import os
import re
import subprocess
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('ipynb_path', type=str)
args = parser.parse_args()

date='2023-12-03'
header = f"""---
title: 'Portfolio Optimization'
date: {date}
permalink: /posts/2023/12/portfolio-optimization/
tags:
  - portfolio
  - pandas
  - stocks
---
"""

'''
Input paths
'''
this_script_dir: str = os.path.abspath(pathlib.Path(__file__).parent.resolve())
ipynb_file_name: str = os.path.basename(args.ipynb_path)

print(f"this dir: {this_script_dir}")
print(f"ipynb file name: {ipynb_file_name}")

# go up the directory two times
root_dir = os.path.abspath(os.path.join(this_script_dir, ".."))
# get posts dir
post_dir = os.path.abspath(os.path.join(root_dir, "_posts"))
#get date from ipynb file
# ipynb_file_name has format YYYY-MM-DD-post-name.ipynb get YYYY-MM-DD

out_dir = post_dir + '/' + date+'-' + ipynb_file_name.split('.')[0]
# date is list of 3 elements, YYYY, MM, DD join them together

print(f"out dir: {out_dir}")

# get markdown file name
md_file_name = date+'-' + ipynb_file_name.split('.')[0] + '.md'
# create markdown 

image_files =   date+'-' + ipynb_file_name.split('.')[0] + '_files'
print(f"name blog: {image_files}")

print(f"md file name: {md_file_name}")

# create folder give by location out_dir
os.makedirs(out_dir, exist_ok=True)

# run the conversion
subprocess.run(['jupyter','nbconvert','--to', 'markdown', '--output-dir', f'{out_dir}', '--output', f'{md_file_name}',f'{args.ipynb_path}'])

# # move image files
if os.path.exists(f'{root_dir}/assets/images/{image_files}/'):
  subprocess.run(['rm','-r',f'{root_dir}/assets/images/{image_files}/'])

subprocess.run(['mv',f'{out_dir}/{image_files}',f'{root_dir}/assets/images/'])


print(f"image files: {image_files}")

# read markdown
with open(os.path.join(out_dir, md_file_name), 'r') as fin:
    md = fin.read()

    print(image_files)
    # replace image files
    modified_content1 = md.replace(f'{image_files}/', f'/assets/images/{image_files}/')

    
    modified_content = modified_content1.replace(r'$', r'$$')

    
    # write markdown
    with open(os.path.join(out_dir, md_file_name), 'w') as fout:
        
        fout.write(header)
        fout.write(modified_content)
