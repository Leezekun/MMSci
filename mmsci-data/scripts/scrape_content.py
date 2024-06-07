import os
import requests
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse
import threading
from subjects import subjects
from utils import *


def scrape_subject(base_path, category, subject, scrape_pdf):

    subject_ = "-".join(subject.lower().split())
    file_path = os.path.join(base_path, category, f"{subject_}_article_links.txt")
    with open(file_path, 'r') as file:
        article_links = file.read().splitlines()

    for url in tqdm(article_links):

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the unique ID from the PDF link
            pdf_link_tag = soup.find('a', href=lambda href: href and '.pdf' in href)
            if pdf_link_tag and pdf_link_tag.get('href'):
                pdf_href = pdf_link_tag['href']
                unique_id = pdf_href.split('/')[-1].replace('.pdf', '')
                pdf_url = f'https://www.nature.com{pdf_href}'
            else:
                unique_id = "unknown"
                pdf_url = "URL not found"
                continue
            
            # Record whether have scraped each kind of information
            scraped_images = scraped_pdf = scraped_title = scraped_time = scraped_abstract \
                = scraped_sections = scraped_reviews = scraped_references = False 
            
            # Determine the path for saving images and JSON
            save_path = os.path.join(base_path, category, subject, unique_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                # Save the final output in a JSON file
                json_filename = os.path.join(save_path, f'{unique_id}_data.json')
                if os.path.exists(json_filename):
                    with open(json_filename, 'r', encoding='utf-8') as json_file:
                        final_output = json.load(json_file)
                else:
                    final_output = {}

                if "images" in final_output:
                    scraped_images = True
                if "pdf_link" in final_output:
                    scraped_pdf = True
                if "title" in final_output:
                    scraped_title = True
                if "published_time" in final_output:
                    scraped_time = True
                if "abstract" in final_output:
                    scraped_abstract = True
                if "sections" in final_output:
                    scraped_sections = True
                if "review_pdf_link" in final_output:
                    scraped_reviews = True
                if "references" in final_output:
                    scraped_references = True

            ################################################
            ##                Scrape Figures              ##
            ################################################
            if not scraped_images:
                images_data = []
                # Extract and download the source URLs of the figures
                figures = soup.find_all('figure')
                for i, figure in enumerate(figures):
                    image_info = {}
                    if figure.find('img'):
                        img_tag = figure.find('img')
                        figcaption = figure.find('figcaption')
                        caption = figcaption.get_text(strip=True) if figcaption else "No caption available"
                        description = figure.p.get_text(strip=True) if figure.p else "No description available"

                        # Image URL
                        if img_tag and img_tag.get('src'):
                            img_url = img_tag['src']
                            if not img_url.startswith('http'):
                                img_url = 'https:' + img_url

                            # Download and save the image
                            img_response = requests.get(img_url)
                            if img_response.status_code == 200:
                                img_filename = f'figure_{i}.png'
                                img_file_path = os.path.join(save_path, img_filename)
                                with open(img_file_path, 'wb') as file:
                                    file.write(img_response.content)

                                # Save the caption and description in a txt file
                                txt_filename = f'figure_{i}_info.txt'
                                txt_file_path = os.path.join(save_path, txt_filename)
                                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                                    txt_file.write(f"Caption: {caption}\nDescription: {description}")

                                # Append image data to the list
                                image_info = {
                                    'image_filename': img_filename,
                                    'text_filename': txt_filename,
                                    'caption': caption,
                                    'description': description
                                }
                                images_data.append(image_info)
                            else:
                                print(f'Failed to download image {i}. Status code: {img_response.status_code}')
                    else:
                        print(f'No image found in figure {i}')
            else:
                images_data = final_output["images"]
            
            ################################################
            ##                Scrape Title                ##
            ################################################
            if not scraped_title:
                title_element = soup.find(class_='c-article-title')
                # Extract the text content of the title element
                if title_element:
                    article_title = title_element.get_text(strip=True)
                    print(f"Article Title: {article_title}")
                else:
                    article_title = ""
                    print("Article title not found.")
            else:
                article_title = final_output["title"]

            ################################################
            ##             Scrape Published Time          ##
            ################################################
            if not scraped_time:
                time_tag = soup.find('time')
                # Extract the 'datetime' attribute value from the <time> tag
                if time_tag:
                    published_time = time_tag['datetime'] 
                    print(f"Published time: {published_time}")
                else:
                    published_time = ""
                    print("Published time not found.")
            else:
                published_time = final_output["published_time"]
            
            ################################################
            ##                Scrape Abstract              ##
            ################################################
            if not scraped_abstract:
                content_div = soup.find('div', class_='c-article-section__content', id='Abs1-content')
                if content_div:
                    abstract = content_div.get_text(separator=' ', strip=True)
                    print(f"Abstract: {abstract}")
                else:
                    abstract = ""
                    print('Abstract not found')
            else:
                abstract = final_output["abstract"]

            ################################################
            ##       Scrape Each section in Main Body     ##
            ################################################
            if not scraped_sections:
                sections = []
                article_body = soup.find('div', class_='c-article-body')
                if article_body:
                    # Find the main-content <div> within the article body
                    main_content = article_body.find('div', class_='main-content')
                    
                    if main_content:
                        # Extract each <section> within the main content
                        html_sections = main_content.find_all('section')
                        
                        for html_section in html_sections:
                            # Get the 'data-title' attribute of each section
                            data_title = html_section.get('data-title', 'No Title')
                            
                            sup_tags = html_section.find_all('sup')
                            for sup_tag in sup_tags:
                                # Find all <a> tags within the <sup> tag that have the 'data-track="click"' attribute
                                a_tags = sup_tag.find_all('a', attrs={'data-track': 'click'})
                                for a_tag in a_tags:
                                    if a_tag.text.isdigit():  # Check if the <a> tag text is a digit
                                        # Replace the <a> tag's text with [number]
                                        new_tag = soup.new_tag("a", attrs=a_tag.attrs)
                                        new_tag.string = f'[{a_tag.text}]'
                                        a_tag.replace_with(new_tag)
                            
                            # After modifications, you can now access the text of each section
                            sec_content = html_section.get_text(separator=' ', strip=True)
                            if sec_content.startswith(data_title):
                                sec_content = sec_content.split(data_title,1)[1].strip()

                            # Print the section's data-title and its text content
                            # print(f"Data Title: {data_title}")
                            # print(sec_content)
                            # print('---' * 20)  # Separator for readability
                            sections.append({"section": data_title, "content": sec_content})
                    else:
                        print("Could not find the 'main-content'.")
                else:
                    print("Could not find the 'c-article-body'.")
            else:
                sections = final_output["sections"]

            
            ################################################
            ##              Scrape Peer review            ##
            ################################################
            if not scraped_reviews:
                # Find the <a> tag with the specific class and data-track-label
                link_tag = soup.find('a', class_='print-link', attrs={'data-track-label': 'peer review file'})
                
                if link_tag:
                    # Step 3: Extract the URL from the href attribute
                    review_pdf_url = link_tag.get('href')
                    
                    # Print or download the PDF from the URL
                    print("PDF URL:", review_pdf_url)
                else:
                    review_pdf_url = "URL not found"
                    print("The 'Peer Review File' link could not be found.")
            else:
                review_pdf_url = final_output["review_pdf_link"]

            if scrape_pdf and review_pdf_url != "URL not found":
                review_pdf_filename = os.path.join(save_path, f'{unique_id}_peer_review_file.pdf')
                if not os.path.exists(review_pdf_filename):
                    pdf_response = requests.get(review_pdf_url)
                    if pdf_response.status_code == 200:
                        with open(review_pdf_filename, 'wb') as f:
                            f.write(pdf_response.content)
                        print("PDF file has been downloaded successfully.")
                    else:
                        print("Failed to download the PDF file.")
            
            if not scraped_references:
                # Find all <li> elements that are classed as article reference items
                reference_items = soup.find_all('li', class_='c-article-references__item js-c-reading-companion-references-item')
                
                references = []  # To store the extracted references
                for item in reference_items:
                    data_counter = item.get('data-counter')

                    # Extract the <p> tag containing the reference text
                    ref_text_tag = item.find('p', class_='c-article-references__text')
                    
                    # Extract the <a> tag with `data-track-action="article reference"` that contains the href
                    ref_link_tag = item.find('a', attrs={'data-track-action': 'article reference'})
                    
                    if ref_text_tag and ref_link_tag:
                        # Get the text and href
                        ref_text = ref_text_tag.get_text(strip=True)
                        ref_href = ref_link_tag.get('href')
                        
                        reference = {"idx": data_counter, "title": ref_text, "link": ref_href}
                        references.append(reference)
                        # print(len(reference))
            else:
                references = final_output["references"]

            ################################################
            ##                Save the data               ##
            ################################################
            if not all([scraped_pdf, scraped_images, scraped_abstract, scraped_title, \
                        scraped_reviews, scraped_time, scraped_sections, scraped_references]):
                
                final_output = {
                    'pdf_link': pdf_url,
                    'review_pdf_link': review_pdf_url,
                    'unique_id': unique_id,
                    'images': images_data,
                    'title': article_title,
                    'published_time': published_time,
                    "abstract": abstract,
                    "sections": sections,
                    "references": references
                }

                json_filename = os.path.join(save_path, f'{unique_id}_data.json')
                # Save the final output in a JSON file
                with open(json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(final_output, json_file, indent=4, ensure_ascii=False)
                print("All data are successfully downloaded at", json_filename)

        else:
            print(f"Failed to fetch the webpage, status code: {response.status_code}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default="all") #
    parser.add_argument('--scrape_pdf', type=str2bool, default=False, help='whether to scrape images', choices=[False, True]) #
    args, unknown = parser.parse_known_args()

    base_path = "../rawdata"
    all_categories = list(subjects.keys())
    if args.category == "all":
        scraped_categories = all_categories
    else:
        assert args.category in all_categories
        scraped_categories = [args.category]


    for category in scraped_categories:
        # Multi-thread
        threads = []
        for subject in subjects[category]:
            thread = threading.Thread(target=scrape_subject, args=(base_path, category, subject, args.scrape_pdf))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        print(f"Scraping completed for all data in category {category}.")
