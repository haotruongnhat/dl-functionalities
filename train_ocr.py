import notebook_utils as nutils
from train import *

xml_paths = nutils.list_files("OCR", "**/*.xml")
dataset = output_set(xml_paths)

classes = ['552F', '550SF','550PF', '552SF', '551F', '551GPF', '522F']

dicts=get_dicts(dataset['train'], classes)

train(dataset, 'ocr_spec', classes)