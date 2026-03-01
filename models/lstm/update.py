import tempfile
import zipfile
import shutil
import os
import re

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='Bury Model Fixer',
                    description='Trains models on generated data')
    parser.add_argument('input', type=str, help="Path to a folder containing folders of models with specific input lengths")
    parser.add_argument('output', type=str, help="Output folder for models")
    
    args = parser.parse_args()
    
    for dir in os.listdir(args.input):
        input_dir = str(os.path.join(args.input, dir))
        if os.path.isdir(input_dir):
            for filename in os.listdir(input_dir):
                input_file = os.path.join(input_dir, filename)
                if os.path.isfile(input_file) and filename.endswith(".keras"):
                    
                    print("Modifying", filename)
                    tempdir = tempfile.mkdtemp()
                    files: dict[str, bytes] = dict()
                    output_file = os.path.join(args.output, dir, filename)
                    os.makedirs(os.path.join(args.output, dir), exist_ok=True)
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    try:
                        tempname = os.path.join(tempdir, filename)
                        with zipfile.ZipFile(input_file, 'r') as zipread:
                            with zipfile.ZipFile(tempname, 'w') as zipwrite:
                                for item in zipread.infolist():
                                    if item.filename not in ["config.json", "metadata.json"]:
                                        data = zipread.read(item.filename)
                                        zipwrite.writestr(item, data)
                                    else:
                                        files[item.filename] = zipread.read(item.filename)
                        shutil.move(tempname, output_file)
                    finally:
                        shutil.rmtree(tempdir)
                    
                    config = re.sub(rb'"float32", "batch_input_shape": \[null, \d+, 1\],', rb'"float32",', files["config.json"]) \
                        .replace(b'"time_major": false, ', b'') \
                        .replace(b'keras.optimizers.legacy', b'keras.optimizers') \
                        .replace(b'"steps_per_execution": null', b'"steps_per_execution": 1')
                    metadata = files["metadata.json"].replace(b"2.15.0", b"3.13.2")
                    with zipfile.ZipFile(output_file, 'a') as z:
                        z.writestr("config.json", config)
                        z.writestr("metadata.json", metadata)
                