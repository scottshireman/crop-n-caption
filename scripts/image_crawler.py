"""
Copyright [2022-2023] Scott Shireman

Licensed under the GNU Affero General Public License;
You may not use this code except in compliance with the License.
You may obtain a copy of the License at

    https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import time
import requests
from bs4 import BeautifulSoup

from queue import Queue
import shutil

def get_args(**parser_kwargs):
    """Get command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to crawl"
        )
    parser.add_argument(
        "--ignore_thumbs",
        type=str,
        default=None,
        help="Ignore URL with 'thumb' in the URL"
        )

    args = parser.parse_args()

    return args

def crawl_url(url, queue, visited_urls, args):
    
    # Download and parse the webpage at the specified URL
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    # Find and download any images on the page.
    image_links = soup.find_all(name='img')

    for link in image_links:
        try:
            image_url = link['src']
        except:
            print(f"not a valid link: {link}")
        else:
            if image_url[0] == "/":
                image_url = url + image_url[1:]

            download_image(image_url, args)


    # Find all the URLs that haven't been visited befoere and add them to the queue
    print(f"Getting URLs from {url} . . .")
    for link in soup.find_all('a'):
        linked_url = link.get('href')

        if "." in linked_url:
            if linked_url.split(".")[1] == "html" and url[-4:] != "html":
                linked_url = url + linked_url
            
        if linked_url[0] == "/":
            linked_url = url + linked_url[1:]

        if url in linked_url and linked_url not in visited_urls:
            # print(f"Put {linked_url}")
            print(f"\tQueueing {linked_url}")
            queue.put(linked_url)


def get_images(page, url, args):

    print(f"Getting images from {url}")
    soup = BeautifulSoup(page)
    image_links = soup.find_all(name='img')

    for link in image_links:
        try:
            image_url = link['src']
        except:
            print(f"not a valid link: {link}")
        else:
            if image_url[0] == "/":
                image_url = url + image_url[1:]

            download_image(image_url, args)


def download_image (url, args):

    if args.ignore_thumbs is False or "thumb" not in url:
        print(f"\tDownloading {url}")
        path = url.replace(args.url,"")
        path = path.replace("/","_")

        # Remove special characters from file name
        if path.isalnum() is False:
            clean_string = "".join(ch for ch in path if (ch.isalnum() or ch == " " or ch == "_" or ch == "."))
            path = clean_string
        
        r = requests.get(url, stream=True)  #Get request
        if r.status_code == 200:            #200 status code = OK
           with open('output/' + path, 'wb') as f: 
              r.raw.decode_content = True
              shutil.copyfileobj(r.raw, f)
    else:
        print(f"\tSkipping {url}")
          

def main():

    start_time = time.time()
    
    args = get_args()
    urls_processed = 0
    visited_urls = []

    queue = Queue()
    queue.put(args.url)

    while not queue.empty():
        next_url = queue.get()

        if next_url not in visited_urls:
            crawl_url(next_url, queue, visited_urls, args)
            visited_urls.append(next_url)
            
    exec_time = time.time() - start_time
    # print(f"  Processed {files_processed} files in {exec_time} seconds for an average of {exec_time/files_processed} sec/file.")
    

if __name__ == "__main__":


    main()
