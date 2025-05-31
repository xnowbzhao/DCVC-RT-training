# Third-Party Implementation of DCVC-RT Training Code

This is a third-party implementation of the training code for the paper  
**"DCVC-RT: Towards Practical Real-Time Neural Video Compression"**.  
Note: This implementation is **not compatible** with the original version, as it was adapted specifically for **medical image compression**.

## Key Modifications

1. The training input is a sequence of medical images with shape `[8, 1, H, W]`.
2. All components related to `cuda_inference` have been removed.
3. The entropy coder has been replaced with the one from the `compressai` library for improved usability.
4. Several parameters have been modified to suit personal requirements.

If you use this code, please make sure to cite the original repository:  
https://github.com/microsoft/DCVC/
