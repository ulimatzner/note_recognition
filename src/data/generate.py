from itertools import product
from pathlib import Path
from typing import Literal

import partitura as pt
from partitura import load_musicxml
from partitura.score import Note, Score
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent.parent.parent

TEMPLATE_XML = ROOT_DIR / 'data' / 'template' / 'template.xml'
Length = Literal['whole', 'half', 'quarter', 'eighth', '16th', '32nd', '64th', '128th', '256th', 'breve', 'long']

NOTE_NAMES = ['E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5']
LENGTHS: list[Length] = ['whole', 'half', 'quarter', 'eighth']


def generate_xml(note_name: str, length: Length, output_file: Path):
    octave = int(note_name[1:])
    note_name = note_name[0]
    file_string = TEMPLATE_XML.read_text()
    file_string = file_string.replace('{{NOTE_NAME}}', f'{note_name}')
    file_string = file_string.replace('{{OCTAVE}}', f'{octave}')
    file_string = file_string.replace('{{LENGTH}}', f'{length}')
    output_file.write_text(file_string)


def generate_xmls():
    dest_dir = ROOT_DIR / 'data' / 'xml'
    dest_dir.mkdir(exist_ok=True)
    for note_name, length in product(NOTE_NAMES, LENGTHS):
        output_file = dest_dir / f'{note_name}_{length}.xml'
        generate_xml(note_name, length, output_file)


def generate_clean_images():
    for file in (ROOT_DIR / 'data' / 'xml').iterdir():
        if file.suffix != '.xml':
            continue
        score = load_musicxml(file)
        dest_dir = ROOT_DIR / 'data' / 'clean_images'
        dest_dir.mkdir(exist_ok=True)
        output_file = dest_dir / f'{file.stem}.png'
        if output_file.exists():
            continue
        pt.render(score, out=output_file, dpi=300)


def generate_cropped_images():
    WIDTH = 80
    X_OFFSET = 105
    HEIGHT = 140
    Y_OFFSET = 20
    output_dir = ROOT_DIR / 'data' / 'cropped_images'
    output_dir.mkdir(exist_ok=True)
    for file in (ROOT_DIR / 'data' / 'clean_images').iterdir():
        img = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
        crop = img[-HEIGHT - Y_OFFSET:-Y_OFFSET, X_OFFSET:X_OFFSET + WIDTH]
        cv.imwrite(str(output_dir / file.name), crop)


def load_images() -> dict[str, np.ndarray]:
    images = {}
    for file in (ROOT_DIR / 'data' / 'cropped_images').iterdir():
        img = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
        images[file.stem] = img
    return images


if __name__ == '__main__':
    generate_xmls()
    generate_clean_images()
    generate_cropped_images()
