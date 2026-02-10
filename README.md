# Super-Resolution of SOHO/MDI Magnetograms of Solar Active Regions Using SDO/HMI Data and an Attention-Aided Convolutional Neural Network
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15250172.svg)](https://doi.org/10.5281/zenodo.15250172)

## Author
Chunhui Xu, Jason T. L. Wang, Haimin Wang, Haodi Jiang, Qin Li, Yasser Abduallah & Yan Xu 

## Abstract
Image super-resolution is an important subject in image processing and recognition. Here, we present an attention-aided convolutional neural network for solar image super-resolution. Our method, named SolarCNN, aims to enhance the quality of line-of-sight (LOS) magnetograms of solar active regions (ARs) collected by the Michelson Doppler Imager (MDI) on board the Solar and Heliospheric Observatory (SOHO). The ground-truth labels used for training SolarCNN are the LOS magnetograms collected by the Helioseismic and Magnetic Imager on board the Solar Dynamics Observatory. Solar ARs consist of strong magnetic fields in which magnetic energy can suddenly be released to produce extreme space-weather events, such as solar flares, coronal mass ejections, and solar energetic particles. SOHO/MDI covers Solar Cycle 23, which is stronger with more eruptive events than Cycle 24. Enhanced SOHO/MDI magnetograms allow for better understanding and forecasting of violent events of space weather. Experimental results show that SolarCNN improves the quality of SOHO/MDI magnetograms in terms of the structural similarity index measure, Pearson's correlation coefficient, and the peak signal-to-noise ratio.

## File Structure

- `train.py`: Training script for the model.
- `test.py`: Testing script to evaluate the model.
- `model.py`: Defines the model architecture.
- `data/image/train/`: Directory containing training data of MDI. Full dataset available via Zenodo.
- `data/image/test/`: Directory containing test data of MDI. A few samples are included; full set available via Zenodo.
- `data/label/train/`: Directory containing training data of HMI. Full dataset available via Zenodo.
- `data/label/test/`: Directory containing test data of HMI. A few samples are included; full set available via Zenodo.
- `data/pred/`: Directory containing predictions. A few samples are included; full set available via Zenodo.
- `model/`: Directory for saving trained models. Download from Zenodo. Moreover, the trained model can also be downloaded directly from [Google Drive](https://drive.google.com/file/d/18YJfFA0VbECbQWTTyr7q-8eOFAktBOOV/view?usp=sharing).
- `prediction_workflow.ipynb`: Workflow of prediction.
- `README.md`: Documentation for setup and usage.

## Environment Setup

Ensure you have Python installed, preferably 3.8+. Install dependencies using:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` with:

```text
tensorflow==2.6
tensorflow-addons
astropy<5.0
protobuf==3.20.3
```

## Training the Model

To train the model, run:

```bash
python train.py
```

This will process the data, train the model, and save it in the `model/` directory.

## Testing the Model

To test a trained model:

```bash
python test.py
```

## Using the Pretrained Model

The trained model is saved in `model/model_solarcnn_retrain`. You can load it in Python as:

```python
import tensorflow as tf
from model import mix_loss, ssim_metric
model = tf.keras.models.load_model("model/model_solarcnn_retrain", custom_objects={'mix_loss': mix_loss, 'ssim_metric': ssim_metric})
```

## References

- Xu, C., Wang, J.T.L., Wang, H. et al. Super-Resolution of SOHO/MDI Magnetograms of Solar Active Regions Using SDO/HMI Data and an Attention-Aided Convolutional Neural Network. Sol Phys 299, 36 (2024). https://doi.org/10.1007/s11207-024-02283-1 [https://link.springer.com/article/10.1007/s11207-024-02283-1](https://link.springer.com/article/10.1007/s11207-024-02283-1)

