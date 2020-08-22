from io import BytesIO
#from urllib.request import urlopen
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import json
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def clean_foldername(address):
  split_address = address.split('/')
  foldername = split_address[-1].split('.')
  return foldername[0]

def clean_filename(address):
  split_address = address.split('/')
  filename = split_address[-1].split('?')
  return filename[0]

def download_file(url, target):
  filename = clean_filename(url)
  if filename not in os.listdir(target):
    print('Downloading ' + filename)
    urlretrieve(url, target + '/' + filename)
  else:
    print(filename + ' already downloaded.')

  return target + '/' + filename

def extract_file(file, target):
    # Download archive
    try:
      with ZipFile(file) as zfile:
        print('Unzipping ' + file)
        zfile.extractall(target)

    except Exception as e:
      print('Failed to unzip ' + file)
      print(e)
      return 1

def dl_unzip(address,target):
  safe_target = os.path.join(os.getcwd(),target)
  foldername = os.path.join(safe_target,clean_foldername(address))
  if not os.path.isdir(safe_target):
    os.makedirs(safe_target)
    print("Creating folder " + target + " in " + os.getcwd())

  file_path = download_file(address, target)
  extract_file(file_path,target)

if __name__ == "__main__":

  with open('config.json') as config_file:
    config = json.load(config_file)

  data_source_train = config['data']['source']['training']
  data_target_train = config['data']['target']['training']
  print("Preparing data sources for training.")
  for data_link in data_source_train:
    dl_unzip(data_link, data_target_train)

  data_source_test = config['data']['source']['testing']
  data_target_test = config['data']['target']['testing']
  print("Preparing data sources for testing.")
  for data_link in data_source_test:
    dl_unzip(data_link, data_target_test)

  print("Installing requirements.")
  requirements = open('requirements.txt', 'r')
  lines = requirements.readlines()

  for package in lines:
    install(package.strip())

  requirements.close()