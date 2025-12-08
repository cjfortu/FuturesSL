"""
Positional encoding modules for the MIGT-TVDT architecture.

Implements composite positional encodings that capture multiple temporal cycles:
- Time-of-day: Sinusoidal encoding for intraday position (ensures 23:55 close to 00:00)
- Day-of-week: Learnable embedding for weekday patterns (Mon-Fri)
- Day-of-month/year: Time2Vec for longer cycles (monthly, seasonal)

Per scientific document Section 3.1, these encodings are added to variable
embeddings before temporal attention to provide temporal context.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class TimeOfDayEncoding(nn.Module):
    """
    Sinusoidal encoding for intraday bar position.
    
    Uses sine/cosine pairs at multiple frequencies to encode position within
    the trading day. The sinusoidal form ensures that bar 287 (23:55) is
    close to bar 0 (00:00) in embedding space, capturing the cyclical nature
    of intraday patterns.
    
    Formula: PE(t, 2i) = sin(2*pi*t / 288 * freq_i)
             PE(t, 2i+1) = cos(2*pi*t / 288 * freq_i)
    
    where freq_i increases geometrically to capture patterns at different scales.
    """
    
    def __init__(self, d_model: int):
        """
        Initialize time-of-day encoding.
        
        Args:
            d_model: Encoding dimension (will use d_model/2 frequency pairs)
                Type: int
                Typical value: 32
        """
        super().__init__()
        self.d_model = d_model
        
        # Precompute frequency divisors for efficiency
        # Use geometric progression of frequencies to capture multi-scale patterns
        # Div[i] = 288 / (2^(i * scale_factor)) where scale_factor spreads frequencies
        half_dim = d_model // 2
        
        # Frequencies: from 1 cycle per day to higher frequencies
        # div_term creates denominators that increase exponentially
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32) * 
            (-np.log(10000.0) / half_dim)
        )
        
        # Register as buffer (not trainable, but moves with model to GPU)
        self.register_buffer('div_term', div_term)
        
    def forward(self, bar_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal encoding for bar positions.
        
        The encoding captures intraday cyclical patterns. Bar 0 and bar 287
        will have similar encodings due to the sinusoidal nature.
        
        Args:
            bar_indices: Integer tensor of bar positions within day
                Type: torch.Tensor (int32 or int64)
                Shape: (B, T) where B=batch, T=sequence length (288)
                Values: 0-287 representing 5-min bars in 24h
                
        Returns:
            Positional encoding tensor
                Type: torch.Tensor (float32)
                Shape: (B, T, d_model)
        """
        # Normalize to [0, 2*pi] for one complete cycle per day
        # Shape: (B, T) -> (B, T, 1)
        position = bar_indices.float().unsqueeze(-1)
        
        # Scale by 2*pi/288 for daily cycle, then multiply by frequencies
        # Shape: (B, T, 1) * (half_dim,) -> (B, T, half_dim)
        angles = position * (2 * np.pi / 288.0) * self.div_term
        
        # Interleave sin and cos: [sin_0, cos_0, sin_1, cos_1, ...]
        # Shape: (B, T, d_model)
        pe = torch.zeros(*bar_indices.shape, self.d_model, device=bar_indices.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)
        
        return pe


class DayOfWeekEncoding(nn.Module):
    """
    Learnable embedding for day of week.
    
    Markets exhibit different behaviors on different weekdays:
    - Monday: Weekend gap effects, position adjustments
    - Friday: Options expiration, reduced afternoon liquidity
    - Mid-week: Generally higher liquidity
    
    Using learnable embeddings allows the model to discover these patterns
    from data rather than imposing assumptions.
    """
    
    def __init__(self, d_model: int):
        """
        Initialize day-of-week embedding.
        
        Args:
            d_model: Embedding dimension
                Type: int
                Typical value: 16
        """
        super().__init__()
        self.d_model = d_model
        
        # 5 trading days: Monday=0 through Friday=4
        # Weekend days (5, 6) shouldn't appear but included for safety
        self.embedding = nn.Embedding(num_embeddings=7, embedding_dim=d_model)
        
    def forward(self, day_indices: torch.Tensor) -> torch.Tensor:
        """
        Look up day-of-week embeddings.
        
        Args:
            day_indices: Day of week indices
                Type: torch.Tensor (int32 or int64)
                Shape: (B,) where B=batch size
                Values: 0=Monday, 1=Tuesday, ..., 4=Friday
                
        Returns:
            Day-of-week embeddings
                Type: torch.Tensor (float32)
                Shape: (B, d_model)
        """
        return self.embedding(day_indices)


class Time2VecEncoding(nn.Module):
    """
    Time2Vec learnable periodic encoding (Kazemi et al., 2019).
    
    Learns both linear and periodic components:
    - Linear term: Captures monotonic trends (T2V[0] = w0*t + b0)
    - Periodic terms: Learn non-obvious cycles (T2V[i] = sin(wi*t + bi))
    
    This is more flexible than fixed sinusoids as it learns the frequencies
    and phases that matter for the specific prediction task. Useful for
    capturing monthly (rebalancing), quarterly (earnings), and seasonal patterns.
    """
    
    def __init__(self, d_model: int, n_periodic: int = None):
        """
        Initialize Time2Vec encoding.
        
        Args:
            d_model: Total encoding dimension
                Type: int
                Typical value: 16-32
            n_periodic: Number of periodic (sinusoidal) components
                Type: int or None
                Default: d_model - 1 (one linear term)
        """
        super().__init__()
        self.d_model = d_model
        self.n_periodic = n_periodic if n_periodic is not None else d_model - 1
        
        # Learnable frequencies and phases
        # Linear term: w0*t + b0
        self.linear_weight = nn.Parameter(torch.randn(1) * 0.01)
        self.linear_bias = nn.Parameter(torch.zeros(1))
        
        # Periodic terms: sin(wi*t + bi) for i in [1, n_periodic]
        # Initialize frequencies to span different scales
        self.periodic_weights = nn.Parameter(
            torch.randn(self.n_periodic) * 0.1
        )
        self.periodic_biases = nn.Parameter(
            torch.zeros(self.n_periodic)
        )
        
    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        Compute Time2Vec encoding.
        
        Args:
            time_values: Normalized time values
                Type: torch.Tensor (float32 or int32/64)
                Shape: (B,) for scalar time (e.g., day_of_year)
                       or (B, T) for sequence of times
                Values: Typically day_of_year (1-366) or day_of_month (1-31)
                
        Returns:
            Time2Vec encoding
                Type: torch.Tensor (float32)
                Shape: (B, d_model) if input is (B,)
                       (B, T, d_model) if input is (B, T)
        """
        # Convert to float and normalize to reasonable range
        t = time_values.float()
        
        # Normalize to [0, 1] range for stability
        # For day_of_year: divide by 366
        # For day_of_month: divide by 31
        # We use 366 as a general normalizer
        t = t / 366.0
        
        # Handle both (B,) and (B, T) input shapes
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Linear term: (B, T, 1)
        linear = self.linear_weight * t.unsqueeze(-1) + self.linear_bias
        
        # Periodic terms: (B, T, n_periodic)
        # Expand t for broadcasting: (B, T, 1) * (n_periodic,) -> (B, T, n_periodic)
        periodic_input = t.unsqueeze(-1) * self.periodic_weights + self.periodic_biases
        periodic = torch.sin(periodic_input)
        
        # Concatenate: (B, T, 1 + n_periodic) = (B, T, d_model)
        output = torch.cat([linear, periodic], dim=-1)
        
        if squeeze_output:
            output = output.squeeze(1)  # (B, d_model)
            
        return output


class CompositePositionalEncoding(nn.Module):
    """
    Combines all positional encodings into unified representation.
    
    Per scientific document Section 3.1, positional encoding captures:
    1. Time-of-day: Intraday patterns (sinusoidal, dim=32)
    2. Day-of-week: Weekly patterns (learnable, dim=16)
    3. Day-of-month: Monthly cycles (Time2Vec, dim=16)
    4. Day-of-year: Seasonal patterns (Time2Vec, dim=32)
    
    Total dimension: 32 + 16 + 16 + 32 = 96, projected to d_model for addition
    to variable embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize composite positional encoding.
        
        Args:
            config: Positional encoding configuration dict
                Type: Dict[str, Any]
                Required keys:
                    time_of_day: {dim: int}
                    day_of_week: {dim: int}
                    day_of_month: {dim: int}
                    day_of_year: {dim: int}
                    d_model: int (target dimension for projection)
        """
        super().__init__()
        
        # Extract dimensions from config
        tod_dim = config['time_of_day']['dim']
        dow_dim = config['day_of_week']['dim']
        dom_dim = config['day_of_month']['dim']
        doy_dim = config['day_of_year']['dim']
        d_model = config['d_model']
        
        # Initialize component encodings
        self.time_of_day = TimeOfDayEncoding(tod_dim)
        self.day_of_week = DayOfWeekEncoding(dow_dim)
        self.day_of_month = Time2VecEncoding(dom_dim)
        self.day_of_year = Time2VecEncoding(doy_dim)
        
        # Projection to model dimension
        total_dim = tod_dim + dow_dim + dom_dim + doy_dim
        self.projection = nn.Linear(total_dim, d_model)
        
        # Store dimensions for reference
        self.total_dim = total_dim
        self.d_model = d_model
        
    def forward(
        self,
        bar_in_day: torch.Tensor,
        day_of_week: torch.Tensor,
        day_of_month: torch.Tensor,
        day_of_year: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined positional encoding.
        
        All encodings are computed and concatenated, then projected to d_model.
        The result is shaped for broadcasting to 4D variable embeddings.
        
        Args:
            bar_in_day: Bar position within day
                Type: torch.Tensor (int32)
                Shape: (B, T) where T=288 (padded sequence length)
                Values: 0-287
            day_of_week: Day of week
                Type: torch.Tensor (int32)
                Shape: (B,)
                Values: 0=Monday, ..., 4=Friday
            day_of_month: Day of month
                Type: torch.Tensor (int32)
                Shape: (B,)
                Values: 1-31
            day_of_year: Day of year
                Type: torch.Tensor (int32)
                Shape: (B,)
                Values: 1-366
                
        Returns:
            Combined positional encoding
                Type: torch.Tensor (float32)
                Shape: (B, T, d_model)
                
        Note:
            Caller must unsqueeze(2) before adding to 4D variable embeddings
            to enable proper broadcasting: (B, T, d_model) -> (B, T, 1, d_model)
        """
        B, T = bar_in_day.shape
        
        # Time-of-day: (B, T, tod_dim)
        tod_enc = self.time_of_day(bar_in_day)
        
        # Day-of-week: (B, dow_dim) -> expand to (B, T, dow_dim)
        dow_enc = self.day_of_week(day_of_week)
        dow_enc = dow_enc.unsqueeze(1).expand(-1, T, -1)
        
        # Day-of-month: (B, dom_dim) -> expand to (B, T, dom_dim)
        dom_enc = self.day_of_month(day_of_month)
        dom_enc = dom_enc.unsqueeze(1).expand(-1, T, -1)
        
        # Day-of-year: (B, doy_dim) -> expand to (B, T, doy_dim)
        doy_enc = self.day_of_year(day_of_year)
        doy_enc = doy_enc.unsqueeze(1).expand(-1, T, -1)
        
        # Concatenate all: (B, T, total_dim)
        combined = torch.cat([tod_enc, dow_enc, dom_enc, doy_enc], dim=-1)
        
        # Project to d_model: (B, T, d_model)
        output = self.projection(combined)
        
        return output
