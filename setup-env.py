from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import os

def clean_foldername(address):
  split_address = address.split('/')
  foldername = split_address[-1].split('.')
  return foldername[0]


def download_file(url,target):
    # Download archive
    try:
      with urlopen(url) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
          zfile.extractall(target)

    except Exception as e:
      print(e)
      return 1

def dl_unzip(address,target):
  safe_target = os.path.join(os.getcwd(),target)
  foldername = os.path.join(safe_target,clean_foldername(address))
  if not os.path.isdir(safe_target):
    os.makedirs(safe_target)
    print("Creating folder " + target + " in " + os.getcwd())

  if not os.path.isdir(foldername):
    print("Downloading and unpacking " + address + " to " + foldername)
    download_file(address,foldername)
  else:
    print(foldername + " already exists.")