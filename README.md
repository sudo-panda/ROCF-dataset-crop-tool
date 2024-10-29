# ROCF dataset crop tool

This tool was created to easily crop the ROCF raw datasets. It uses a pdf of raw ROCF sketch images and then extract the images and suggests a cropping window. The user can then fix any wrong crop suggestions. It optionally allows you to specify an excel file to be able to add patient number and ROCF test type in the name of the croppped image when it is saved.

## Usage

Just install the required packages and run the python file
```
pip install -r requirements.txt
python crop_rocf.py
```