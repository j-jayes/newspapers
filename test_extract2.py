import logging
logging.basicConfig(level=logging.INFO)
from newspapers.extraction.extract import extract_job_ad_with_grounding

def main():
    text = "Sökes 2ne ordentliga Springgossar i 15-års åldern för anställning vid tryckeri. Lön 4 kr/vecka. Svante Johansson."
    print("Testing extraction on short text...")
    ad, doc = extract_job_ad_with_grounding(text, "test.txt")
    print(ad)

if __name__ == '__main__':
    main()
