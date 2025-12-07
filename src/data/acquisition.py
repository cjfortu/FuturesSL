"""
NQ Futures Data Acquisition Module
==================================

This module handles downloading NASDAQ 100 E-mini futures (NQ) OHLCV data from
Databento's historical data API. It implements chunked downloading for large date
ranges, retry logic for transient failures, and validation of downloaded data.

Key Design Decisions:
- Uses NQ.v.0 (volume-based rollover) per problem statement requirement
- Chunks requests into 90-day segments to avoid API timeouts
- Validates OHLC relationships and price sanity
- Prices are NOT scaled after to_df() as Databento already converts fixed-point to floats

References:
    - grok-scientific.md Section 2.1: Volume-based rollover requirement
    - claude-engineering.md Section 3.1.1: Databento configuration
    - Databento API docs: https://databento.com/docs

Authors: Claude (Engineering Lead), Gemini (Research), Grok (Scientific)
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


class DatabentoConfig:
    """
    Configuration constants for Databento API access.
    
    Per claude-engineering.md Section 3.1.1:
    - NQ.v.0 = Volume-based rollover (required by problem statement)
    - NQ.c.0 = Calendar-based rollover (NOT used)
    - NQ.n.0 = Open interest-based rollover (NOT used)
    
    Attributes:
        API_KEY: Databento API key for authentication.
        DATASET: CME Globex MDP3 dataset identifier.
        SYMBOL: NQ continuous contract with volume-based rollover.
        SCHEMA: 1-minute OHLCV bar data.
        START_DATE: Beginning of data range (June 2010).
        END_DATE: End of data range (December 2025).
    """
    
    API_KEY: str = "db-TwwdiEYy786bMK6Y4ahnuPPYkJVcg"
    DATASET: str = "GLBX.MDP3"
    SYMBOL: str = "NQ.v.0"  # Volume-based continuous contract
    SCHEMA: str = "ohlcv-1m"
    START_DATE: str = "2010-06-06"
    END_DATE: str = "2025-12-03"


class NQDataAcquisition:
    """
    Handles data acquisition from Databento for NQ futures.
    
    This class manages the complete data download pipeline including:
    - Cost estimation before download
    - Chunked downloading for large date ranges
    - Retry logic for transient API errors
    - Data validation and sanity checks
    - Parquet storage with compression
    
    CRITICAL NOTE:
        Databento's to_df() method automatically converts fixed-point integer
        prices to floats. Do NOT apply additional scaling (e.g., multiplying
        by 1e-9) as this would corrupt the data.
    
    Attributes:
        client: Databento Historical API client instance.
        output_dir: Directory for saving downloaded data.
        chunk_days: Number of days per API request chunk.
        max_retries: Maximum retry attempts for failed requests.
        retry_delay: Base delay between retries in seconds.
    
    Example:
        >>> acquisition = NQDataAcquisition(api_key="db-xxx", output_dir="./data")
        >>> cost = acquisition.estimate_cost("2020-01-01", "2020-12-31")
        >>> df = acquisition.download_range("2020-01-01", "2020-12-31")
        >>> acquisition.save_parquet(df, "nq_2020.parquet")
    """
    
    # Class constants for chunking and retry behavior
    DEFAULT_CHUNK_DAYS: int = 90
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_RETRY_DELAY: int = 5  # seconds
    
    def __init__(
        self,
        api_key: str,
        output_dir: str,
        chunk_days: int = DEFAULT_CHUNK_DAYS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: int = DEFAULT_RETRY_DELAY
    ):
        """
        Initialize the NQ data acquisition handler.
        
        Args:
            api_key: Databento API key for authentication.
                Type: str
                Format: "db-XXXX..." (Databento key format)
            output_dir: Directory path for saving downloaded data.
                Type: str
                Will be created if it doesn't exist.
            chunk_days: Number of days per API request chunk.
                Type: int
                Default: 90 (prevents API timeouts)
            max_retries: Maximum retry attempts for failed requests.
                Type: int
                Default: 3
            retry_delay: Base delay between retries in seconds.
                Type: int
                Default: 5 (multiplied by attempt number)
        
        Raises:
            ImportError: If databento package is not installed.
        """
        try:
            import databento as db
            self.client = db.Historical(api_key)
        except ImportError:
            raise ImportError(
                "databento package not installed. Install with: pip install databento"
            )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_days = chunk_days
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"NQDataAcquisition initialized. Output dir: {self.output_dir}")
    
    def estimate_cost(
        self,
        start: str,
        end: str,
        symbol: str = DatabentoConfig.SYMBOL,
        schema: str = DatabentoConfig.SCHEMA
    ) -> float:
        """
        Estimate download cost before committing to API charges.
        
        This method queries Databento's metadata API to get the estimated cost
        for downloading the specified data range. Always run this before
        downloading to verify budget (~$100-500 for full 15-year range).
        
        Args:
            start: Start date in YYYY-MM-DD format.
                Type: str
                Example: "2010-06-06"
            end: End date in YYYY-MM-DD format.
                Type: str
                Example: "2025-12-03"
            symbol: Databento symbol identifier.
                Type: str
                Default: "NQ.v.0" (volume-based continuous)
            schema: Data schema type.
                Type: str
                Default: "ohlcv-1m" (1-minute bars)
        
        Returns:
            Estimated cost in USD as a float.
            Type: float
            Example: 150.25
        
        Raises:
            Exception: If API call fails (network, authentication, etc.)
        """
        cost = self.client.metadata.get_cost(
            dataset=DatabentoConfig.DATASET,
            symbols=[symbol],
            stype_in="continuous",
            schema=schema,
            start=start,
            end=end,
        )
        
        logger.info(f"Estimated cost for {start} to {end}: ${cost:.2f}")
        return cost
    
    def download_range(
        self,
        start: str,
        end: str,
        symbol: str = DatabentoConfig.SYMBOL,
        schema: str = DatabentoConfig.SCHEMA,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Download OHLCV data for the specified date range.
        
        Downloads data in chunks to handle large date ranges without API timeouts.
        Implements exponential backoff retry logic for transient failures.
        
        CRITICAL: Databento's to_df() already converts fixed-point integers to
        floats. Do NOT apply additional scaling - prices will be in correct
        range (e.g., 20000.00 for NQ).
        
        Args:
            start: Start date in YYYY-MM-DD format.
                Type: str
                Example: "2010-06-06"
            end: End date in YYYY-MM-DD format.
                Type: str
                Example: "2025-12-03"
            symbol: Databento symbol identifier.
                Type: str
                Default: "NQ.v.0" (volume-based continuous)
            schema: Data schema type.
                Type: str
                Default: "ohlcv-1m" (1-minute bars)
            validate: Whether to run validation checks on downloaded data.
                Type: bool
                Default: True
        
        Returns:
            DataFrame with processed OHLCV data.
            Type: pd.DataFrame
            Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            Shape: (num_bars, 6)
        
        Raises:
            ValueError: If no data is retrieved or validation fails.
            Exception: If all retry attempts fail for a chunk.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        
        all_data = []
        current_start = start_dt
        total_bars = 0
        
        logger.info(f"Starting download: {start} to {end}")
        
        while current_start < end_dt:
            current_end = min(
                current_start + timedelta(days=self.chunk_days),
                end_dt
            )
            
            logger.info(f"Downloading chunk: {current_start.date()} to {current_end.date()}")
            
            # Retry loop for transient errors
            chunk_df = self._download_chunk_with_retry(
                current_start, current_end, symbol, schema
            )
            
            if chunk_df is not None and len(chunk_df) > 0:
                all_data.append(chunk_df)
                total_bars += len(chunk_df)
                logger.info(f"  Retrieved {len(chunk_df):,} bars (total: {total_bars:,})")
            
            current_start = current_end
        
        if not all_data:
            raise ValueError(f"No data retrieved for range {start} to {end}")
        
        # Concatenate and deduplicate
        df = pd.concat(all_data, ignore_index=False)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # Process into standard format
        processed = self._process_raw_data(df)
        
        # Validate if requested
        if validate:
            self._validate_ohlcv(processed)
        
        logger.info(f"Download complete: {len(processed):,} total bars")
        return processed
    
    def _download_chunk_with_retry(
        self,
        start_dt: datetime,
        end_dt: datetime,
        symbol: str,
        schema: str
    ) -> Optional[pd.DataFrame]:
        """
        Download a single chunk with retry logic.
        
        Implements exponential backoff: delay = retry_delay * attempt_number.
        
        Args:
            start_dt: Chunk start datetime.
                Type: datetime
            end_dt: Chunk end datetime.
                Type: datetime
            symbol: Databento symbol.
                Type: str
            schema: Data schema.
                Type: str
        
        Returns:
            DataFrame from to_df() or None if chunk has no data.
            Type: Optional[pd.DataFrame]
        
        Raises:
            Exception: If all retry attempts fail.
        """
        for attempt in range(self.max_retries):
            try:
                data = self.client.timeseries.get_range(
                    dataset=DatabentoConfig.DATASET,
                    symbols=symbol,
                    stype_in="continuous",
                    schema=schema,
                    start=start_dt.strftime("%Y-%m-%dT00:00:00"),
                    end=end_dt.strftime("%Y-%m-%dT23:59:59"),
                )
                
                # CRITICAL: to_df() already converts fixed-point to floats
                # Prices will be in correct range (e.g., 20000.00 for NQ)
                df = data.to_df()
                return df
                
            except Exception as e:
                logger.warning(
                    f"  Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (attempt + 1)
                    logger.info(f"  Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"  Failed after {self.max_retries} attempts")
                    raise
        
        return None
    
    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Databento DataFrame format to standard OHLCV format.
        
        NOTE: Prices are already floats from to_df() - no scaling needed!
        This method extracts the relevant columns and resets the index.
        
        Args:
            df: Raw DataFrame from Databento's to_df().
                Type: pd.DataFrame
                Expected columns: ['open', 'high', 'low', 'close', 'volume']
                Index: DatetimeIndex
        
        Returns:
            Processed DataFrame with timestamp column.
            Type: pd.DataFrame
            Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            Shape: (num_bars, 6)
        """
        processed = pd.DataFrame()
        
        # Extract timestamp from index
        processed['timestamp'] = df.index
        
        # Copy price columns directly - already floats!
        for col in ['open', 'high', 'low', 'close']:
            processed[col] = df[col].values
        
        # Volume as int64
        processed['volume'] = df['volume'].values.astype(np.int64)
        
        # Reset index for clean DataFrame
        processed = processed.reset_index(drop=True)
        
        return processed
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """
        Validate OHLCV data integrity with detailed reporting.
        
        Performs the following checks:
        1. OHLC relationship validity (high >= low, prices within range)
        2. Zero/negative price detection
        3. Price magnitude sanity check (NQ should be in thousands)
        
        Args:
            df: DataFrame with OHLCV data.
                Type: pd.DataFrame
                Required columns: ['open', 'high', 'low', 'close', 'volume']
        
        Raises:
            ValueError: If price sanity check fails (suggests scaling bug).
        
        Logs warnings for invalid bars but does not raise exceptions for
        minor data quality issues (these may occur in raw market data).
        """
        # Check OHLC relationships
        invalid_hl = df['high'] < df['low']
        invalid_oh = df['open'] > df['high']
        invalid_ol = df['open'] < df['low']
        invalid_ch = df['close'] > df['high']
        invalid_cl = df['close'] < df['low']
        
        total_invalid = (
            invalid_hl | invalid_oh | invalid_ol | invalid_ch | invalid_cl
        ).sum()
        
        if total_invalid > 0:
            pct_invalid = 100 * total_invalid / len(df)
            logger.warning(
                f"Found {total_invalid:,} bars ({pct_invalid:.4f}%) with "
                f"invalid OHLC relationships"
            )
        
        # Check for zero/negative prices
        zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if zero_prices > 0:
            logger.warning(f"Found {zero_prices:,} bars with zero/negative prices")
        
        # Sanity check: NQ should be in thousands range
        median_price = df['close'].median()
        min_price = df['close'].min()
        max_price = df['close'].max()
        
        logger.info(
            f"Price statistics: median={median_price:.2f}, "
            f"min={min_price:.2f}, max={max_price:.2f}"
        )
        
        # NQ has traded between ~1000 (2010) and ~22000+ (2024)
        # If median is below 100, something is wrong
        if median_price < 100:
            raise ValueError(
                f"CRITICAL: Median price {median_price:.6f} is too low. "
                f"This suggests a scaling bug - check that prices are not "
                f"being multiplied by 1e-9 after to_df()."
            )
        
        logger.info("Price sanity check passed")
    
    def save_parquet(
        self,
        df: pd.DataFrame,
        filename: str,
        compression: str = 'snappy'
    ) -> Path:
        """
        Save DataFrame to Parquet format with compression.
        
        Parquet is chosen for:
        - Efficient columnar storage
        - Fast read times for feature engineering
        - Compatibility with pandas, PyArrow, and Spark
        
        Args:
            df: DataFrame to save.
                Type: pd.DataFrame
            filename: Output filename (will be placed in output_dir).
                Type: str
                Example: "nq_ohlcv_1m_raw.parquet"
            compression: Parquet compression algorithm.
                Type: str
                Default: 'snappy' (good balance of speed/size)
                Options: 'snappy', 'gzip', 'brotli', 'zstd', None
        
        Returns:
            Full path to saved file.
            Type: Path
        """
        filepath = self.output_dir / filename
        
        df.to_parquet(
            filepath,
            engine='pyarrow',
            compression=compression,
            index=False
        )
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df):,} rows to {filepath} ({file_size_mb:.1f} MB)")
        
        return filepath


def download_full_dataset(
    output_dir: str,
    api_key: str = DatabentoConfig.API_KEY,
    start_date: str = DatabentoConfig.START_DATE,
    end_date: str = DatabentoConfig.END_DATE,
    dry_run: bool = True
) -> Optional[pd.DataFrame]:
    """
    Convenience function to download the complete NQ dataset.
    
    This function wraps the NQDataAcquisition class for simple one-call
    downloading of the full dataset specified in the problem statement.
    
    Args:
        output_dir: Directory for saving data.
            Type: str
        api_key: Databento API key.
            Type: str
            Default: From DatabentoConfig
        start_date: Start date in YYYY-MM-DD format.
            Type: str
            Default: "2010-06-06"
        end_date: End date in YYYY-MM-DD format.
            Type: str
            Default: "2025-12-03"
        dry_run: If True, only estimate cost without downloading.
            Type: bool
            Default: True (safety default to prevent accidental charges)
    
    Returns:
        Downloaded DataFrame if dry_run=False, None otherwise.
        Type: Optional[pd.DataFrame]
    
    Example:
        >>> # First, check cost
        >>> download_full_dataset("./data", dry_run=True)
        >>> # Then download
        >>> df = download_full_dataset("./data", dry_run=False)
    """
    acquisition = NQDataAcquisition(api_key=api_key, output_dir=output_dir)
    
    # Always estimate cost first
    cost = acquisition.estimate_cost(start_date, end_date)
    print(f"\nEstimated cost: ${cost:.2f}")
    
    if dry_run:
        print("Dry run mode - set dry_run=False to proceed with download")
        return None
    
    # Download and save
    df = acquisition.download_range(start_date, end_date)
    acquisition.save_parquet(df, "nq_ohlcv_1m_raw.parquet")
    
    return df
