import requests
from bs4 import BeautifulSoup
import os
import argparse
import time

from subjects import subjects
from utils import *
    
def scrape_article_links(base_url):
    all_links = set()
    page = 1

    while True:
        url = f"{base_url}?page={page}"
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            article_links = soup.find_all('a', href=lambda href: href and '/articles/' in href)

            # Check if there are no articles on the page, then break
            if not article_links:
                break

            for link in article_links:
                href = link.get('href')
                full_url = f'https://www.nature.com{href}' if not href.startswith('http') else href
                all_links.add(full_url)

            print(f"Scraped page {page}")
            page += 1
        else:
            print(f"Failed to retrieve page {page}. Status code: {response.status_code}")
            break

    return all_links


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument('--category', type=str, default="all") #

    args, unknown = parser.parse_known_args()
    print(args)

    base_path = "../rawdata"
    all_categories = list(subjects.keys())
    if args.category == "all":
        scraped_categories = all_categories
    else:
        assert args.category in all_categories
        scraped_categories = [args.category]

    for category in scraped_categories:
        for subject in subjects[category]:
            subject = "-".join(subject.lower().split())

            # URL of the base page
            base_url = f'https://www.nature.com/subjects/{subject}/ncomms'

            # Call the scraping function
            article_links = scrape_article_links(base_url)

            # Create a new folder for the scraped data
            folder_path = os.path.join(base_path, category)
            os.makedirs(folder_path, exist_ok=True)

            # Save the links to a file in the new folder
            file_path = os.path.join(folder_path, f'{subject}_article_links.txt')
            with open(file_path, 'w') as file:
                for link in article_links:
                    file.write(link + '\n')

            print(f"Category {category}, subject {subject}, total {len(article_links)} links scraped and saved to {file_path}")

    end_time = time.time()
    total_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f"Total running time: {formatted_time}")