import torch
import numpy as np
from typing import Union, Literal, Callable, Optional

def pass_filter_old(tensor: torch.Tensor, type: Literal["low", "high"] = "low", algorithm: Literal["hann", "hamming", "blackman", "kaiser"] = "hann", cutoff_freq: Union[int, float] = 10000, sample_rate: Union[int, float] = 44100, beta: Union[int, float] = 14) -> torch.Tensor:
    """
    Apply a low-pass or high-pass filter to a tensor using one of several algorithms.

    Parameters:
    tensor: The input tensor. This should be a 1D tensor containing the data to be filtered.
    type: The type of filter to apply. This should be either "low" for a low-pass filter or "high" for a high-pass filter.
    algorithm: The algorithm to use for the filter. This should be one of the following: 
        "hann", "hamming", "blackman", "kaiser".
    cutoff_freq: The cutoff frequency for the filter. All frequencies above (for a low-pass filter) or below (for a high-pass filter) this value will be attenuated.
    sample_rate: The sample rate of the data. This is used to convert the cutoff frequency into a corresponding number of samples.
    beta: The beta parameter for the Kaiser window function. 
        This parameter determines the trade-off between the main-lobe width and side-lobe level. 
        A larger value will result in a narrower main-lobe and lower side-lobe level, at the expense of a slower roll-off. 
        This parameter is ignored for other window functions.

    Returns:
    The filtered tensor.
    """
    # Apply window function to the input tensor
    algorithms = {
        "hann": lambda size: torch.hann_window(size, dtype=tensor.dtype, device=tensor.device),
        "hamming": lambda size: torch.hamming_window(size, dtype=tensor.dtype, device=tensor.device),
        "blackman": lambda size: torch.blackman_window(size, dtype=tensor.dtype, device=tensor.device),
        "kaiser": lambda size: torch.kaiser_window(size, beta=beta, dtype=tensor.dtype, device=tensor.device),
        "rectangular": lambda size: torch.ones(size, dtype=tensor.dtype, device=tensor.device)  # добавляем прямоугольное окно
    }
    window = algorithms[algorithm](tensor.numel())
    windowed_tensor = tensor * window

    # Apply Fourier transform
    spectrum = torch.fft.fft(windowed_tensor)

    # Compute frequencies for each component of the spectrum
    frequencies = torch.fft.fftfreq(tensor.numel(), 1.0 / sample_rate)

    # Create a mask for the frequencies we want to keep
    if type == "low":
        mask = torch.abs(frequencies) <= cutoff_freq
    elif type == "high":
        mask = torch.abs(frequencies) > cutoff_freq

    # Apply the mask to the spectrum
    spectrum = spectrum * mask.to(tensor.device)

    # Apply inverse Fourier transform
    filtered_tensor = torch.fft.ifft(spectrum).real
    # Scale the filtered tensor to have a similar average amplitude as the input tensor
    input_amplitude = torch.mean(torch.abs(tensor))
    filtered_tensor = filtered_tensor * (input_amplitude / torch.mean(torch.abs(filtered_tensor)))



    return filtered_tensor

import torch
from typing import Union

import torch
from typing import Union

import torch
from typing import Union

def pass_filter(tensor: torch.Tensor, filter_type: str = "low", cutoff_freq: Union[int, float] = 10000, sample_rate: Union[int, float] = 44100, transition_bandwidth: Union[int, float] = 800) -> torch.Tensor:
    """
    Apply a low-pass or high-pass filter to a tensor.

    Parameters:
    tensor: The input tensor. This should be a 1D tensor containing the data to be filtered.
    filter_type: The type of filter to apply. This should be either "low" for a low-pass filter or "high" for a high-pass filter.
    cutoff_freq: The cutoff frequency for the filter. All frequencies above (for a low-pass filter) or below (for a high-pass filter) this value will be attenuated.
    sample_rate: The sample rate of the data. This is used to convert the cutoff frequency into a corresponding number of samples.
    transition_bandwidth: The width of the transition band for the filter. Frequencies in this range are attenuated with a smooth window function to avoid ringing artifacts.

    Returns:
    The filtered tensor.
    """
    # Apply Fourier transform
    spectrum = torch.fft.fft(tensor)

    # Compute frequencies for each component of the spectrum
    frequencies = torch.fft.fftfreq(tensor.numel(), 1.0 / sample_rate).to(tensor.device)

    # Define the passband and transition band
    if filter_type == "low":
        passband = torch.abs(frequencies) <= cutoff_freq
        transition_band = (torch.abs(frequencies) > cutoff_freq) & (torch.abs(frequencies) <= (cutoff_freq + transition_bandwidth))
    elif filter_type == "high":
        passband = torch.abs(frequencies) >= cutoff_freq
        transition_band = (torch.abs(frequencies) < cutoff_freq) & (torch.abs(frequencies) >= (cutoff_freq - transition_bandwidth))

    # Create a mask for the frequencies we want to keep
    transition_band_elements = transition_band.sum().item()
    if transition_band_elements > 0:
        window = torch.hann_window(transition_band_elements, device=tensor.device).to(tensor.device)
        expanded_transition_band = transition_band.float().clone()
        expanded_transition_band[transition_band] = window
        mask = passband.float().to(tensor.device) + expanded_transition_band
    else:
        mask = passband.to(tensor.device)

    # Apply the mask to the spectrum
    spectrum = spectrum * mask

    # Apply inverse Fourier transform
    filtered_tensor = torch.fft.ifft(spectrum).real

    return filtered_tensor






def filter_chain(filter: Callable=None, **kwargs) -> Callable:
    """
    Create a filter function that can be used to apply a sequence of filters to a tensor.

    Parameters:
    type: The type of filter to apply. This should be either "low" for a low-pass filter or "high" for a high-pass filter.
    algorithm: The algorithm to use for the filter. This should be one of the following: 
        "hann", "hamming", "blackman", "kaiser".
    cutoff_freq: The cutoff frequency for the filter. All frequencies above (for a low-pass filter) or below (for a high-pass filter) this value will be attenuated.
    sample_rate: The sample rate of the data. This is used to convert the cutoff frequency into a corresponding number of samples.
    beta: The beta parameter for the Kaiser window function. 
        This parameter determines the trade-off between the main-lobe width and side-lobe level. 
        A larger value will result in a narrower main-lobe and lower side-lobe level, at the expense of a slower roll-off. 
        This parameter is ignored for other window functions.
    filter: The previous filter function in the chain. If this is None, the new filter will be applied to the input tensor directly.

    Returns:
    The new filter function.
    """
    # Define the new filter function
    def new_filter(tensor: torch.Tensor) -> torch.Tensor:
        # Apply the previous filter function if it exists
        if filter is not None:
            tensor = filter(tensor)

        # Apply the new filter to the tensor
        tensor = pass_filter(tensor, **kwargs)

        return tensor

    return new_filter
