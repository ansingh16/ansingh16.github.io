import os
import re
import subprocess
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('ipynb_path', type=str)
args = parser.parse_args()


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

out_dir = post_dir + '/' + ipynb_file_name.split('.')[0]
# date is list of 3 elements, YYYY, MM, DD join them together

print(f"out dir: {out_dir}")

# get markdown file name
md_file_name = ipynb_file_name.split('.')[0] + '.md'
# create markdown 

image_files = ipynb_file_name.split('.')[0] + '_files'
print(f"name blog: {image_files}")

print(f"md file name: {md_file_name}")

# create folder give by location out_dir
os.makedirs(out_dir, exist_ok=True)

# run the conversion
subprocess.run(['jupyter','nbconvert','--to', 'markdown', '--output-dir', f'{out_dir}', '--output', f'{md_file_name}',f'{args.ipynb_path}'])

# move image files
subprocess.run(['mv',f'{out_dir}/{image_files}',f'{root_dir}/assets/images/'])

# read markdown
with open(os.path.join(out_dir, md_file_name), 'r') as fin:
    md = fin.read()

  
    # replace image files
    modified_content = md.replace(f'{image_files}/', f'/assets/images/{image_files}/')

    # write markdown
    with open(os.path.join(out_dir, md_file_name), 'w') as fout:
        
        fout.write(modified_content)
