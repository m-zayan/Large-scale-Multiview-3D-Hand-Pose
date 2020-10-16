# -*- coding: utf-8 -*-
"""download_zip_file.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CdOJO-UY6UXYJfiLkxQRLaBaHz72G5xC
"""

from typing import Union

import os
from pathlib import Path

import requests

import numpy as np
import cv2

import io
from zipfile import ZipFile

import tensorflow as tf

"""# Method - 1"""

class Writer:

    def __init__(self):

        pass

    @staticmethod
    def download_from_url(url: str, to_path: str, chunk_size: int = 1024):
        
        """
        Parameter
        ---------
        url (str): file url
        to_path (str): director/fname + file_extension
        chunk_size: request chunk_size, default = 1024
        Returns
        -------
        str: encoded string, ex. encoding='utf-8'
        """

        req = requests.get(url, stream=True)

        if req.status_code != 200:
            
            raise req.raise_for_status()

        else:

            with open(to_path, "wb") as file:
               
                for block in req.iter_content(chunk_size=chunk_size):
                    
                    if block:

                        file.write(block)
    
    @staticmethod
    def extractall(path: str, to_path: str):

      with ZipFile(path, 'r') as zipfile_buffer:

          zipfile_buffer.extractall(to_path)

    @staticmethod
    def get_zipfile_content(url):
        
        """
        Parameter
        ---------
        url (str): url for zip-file
        Returns
        -------
        ZipFile: ZipFile Object, of bytes data
        """

        res = requests.get(url, stream=True)

        d_bytes = io.BytesIO(res.content)
        files = ZipFile(d_bytes)

        return files

    @staticmethod
    def get_zipfile_bytes(url, fname):
       
        """
        Parameter
        ---------
        url (str): url for zip-file
        fname (str): A specific file inside zip file (ex. age.csv, cat_id_1.png, ...etc.)
        Returns
        -------
        list: list of bytes contains, file data as in bytes format
        """

        files = Writer.get_zipfile_content(url)
        d_bytes = None

        try:
          
            d_bytes = files.open(fname)
        
        except KeyError as err:

            print(err, '\nAvailable files : \n', '=' * 50,  files.filelist)

        bytes_data = d_bytes.readlines()

        return bytes_data

"""# Method - 2 (Using tensorflow)"""

def download(url, filename, extract=False, cache_subdir='datasets'):
  
  """
  Parameter
  ---------
  url (str): url for zip-file
  fname (str): A specific file inside zip file (ex. age.csv, cat_id_1.png, ...etc.)
  Returns
  -------
  file root directory, and files name
  """

  root = tf.keras.utils.get_file(filename, url, cache_subdir=cache_subdir, extract=extract)
  root = '/'.join(root.split('/')[:-1])

  files_name = os.listdir(root)

  return root, files_name

"""# Reader"""

class Reader:

  def __init__(self):
    
    pass


  @staticmethod
  def read_images(images_dir: str, shape: Union[tuple, list]=None, fix_color=False):
    
    """
    Parameter
    ---------
    
    images_dir (str): url for zip-file

    shape: Union[tuple, list]: if not None, then resize the image with a given shape

    Returns
    -------
    images generator as numpy array
    """

    images_path = os.listdir(images_dir)

    for filename in images_path:
      
      path = os.path.join(images_dir, filename)

      yield Reader.read_image(path, shape, fix_color)

  @staticmethod
  def read_image(path: str, shape: Union[tuple, list]=None, fix_color=False):

    """
    Parameter
    ---------
    path (str): image path

    shape: Union[tuple, list]: if not None, then resize the image with a given shape
    
    Returns
    -------
    image as numpy array
    """

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image is None:

      raise ValueError(f'Unable to read the image : {path}')

    if shape is not None:

      if len(shape) != 2:
        
        raise ValueError(f'Expect 2D shape, and got {len(shape)}D : {shape}')

      image = cv2.resize(image, shape)

    if fix_color:

      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image