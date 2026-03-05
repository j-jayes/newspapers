import logging
logging.basicConfig(level=logging.INFO)
from newspapers.extraction.extract import process_advertisement, transcribe_advertisement_image
from pathlib import Path
from PIL import Image

def main():
    p = list(Path('data/processed').glob('*.jpg'))[0]
    print(f'Testing on {p.name}')
    
    # Just test transcription first to avoid full pipeline hang
    print('Starting transcription...')
    text = transcribe_advertisement_image(p)
    print(f'Transcription complete: {len(text)} characters')
    print('Preview:')
    print(text[:200])

if __name__ == '__main__':
    main()
