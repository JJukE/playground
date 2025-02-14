import os
import argparse
import shutil
from pathlib import Path

if __name__ == "__main__":

    cat_dict = {
        "plane": "02691156",
        "car": "02958343",
        "chair": "03001627",
        "table": "04379243"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to input directory. (Should contain the class id if you want to process the c)')
    args = parser.parse_args()

    # paths_r = list(Path(args.data_dir).glob("**/*r_000.png"))
    # paths_albedo = list(Path(args.data_dir).glob("**/*r_000_albedo0001.png"))
    # paths_depth = list(Path(args.data_dir).glob("**/*r_000_depth0001.png"))
    # paths_normal = list(Path(args.data_dir).glob("**/*r_000_normal0001.png"))
    # print(len(paths_r))
    # print(len(paths_albedo))
    # print(len(paths_depth))
    # print(len(paths_normal))
    
    # paths_temp = list(Path(args.data_dir).glob("**/*model_normalized.obj")) ## for rendering without textures
    # print(len(paths_temp))
    # paths = list(Path(args.data_dir).glob("**/*model_normalized_nomtl.obj"))
    # paths = list(Path(args.data_dir).glob("**/*.npz"))
    # print(len(paths))

    # # # split train, valid, test model_ids and save them.
    # num_trains = int(len(paths) * 0.96)
    # num_valids = (len(paths) - num_trains) // 2
    # num_tests = len(paths) - num_trains - num_valids
    # train_ids = []
    # valid_ids = []
    # test_ids = []
    # for i, path in enumerate(paths):
    #     if i < num_trains:
    #         train_ids.append(path.parts[-3])
    #     elif i < num_trains + num_valids:
    #         valid_ids.append(path.parts[-3])
    #     elif i < num_trains + num_valids + num_tests:
    #         test_ids.append(path.parts[-3])
    #     else:
    #         print("It should not be printed.")
    # print(len(train_ids))
    # print(len(valid_ids))
    # print(len(test_ids))
    # print(len(train_ids) + len(valid_ids) + len(test_ids))

    # with open("/root/hdd2/ShapeNetCoreV2/split/04379243/model_ids_train.txt", "w") as file:
    #     for line in train_ids:
    #         file.write(str(line) + "\n")
    # with open("/root/hdd2/ShapeNetCoreV2/split/04379243/model_ids_valid.txt", "w") as file:
    #     for line in valid_ids:
    #         file.write(str(line) + "\n")
    # with open("/root/hdd2/ShapeNetCoreV2/split/04379243/model_ids_test.txt", "w") as file:
    #     for line in test_ids:
    #         file.write(str(line) + "\n")

    # # looking for missing renderings
    # model_ids = []
    # for path_r in paths_r:
    #     model_ids.append(path_r.parts[-3])
    
    # print(len(model_ids))
    # for path in paths:
    #     if path.parts[-3] not in model_ids:
    #         print(path.parts[-3])
    
    paths_ = list(Path(args.data_dir).glob("**/*volume_500K.npy"))
    print(len(paths_))
    print(paths_[0])
    
    # paths_200k = []
    # paths_500k = []
    # for path in paths:
    #     if "500K" in str(path):
    #         paths_500k.append(path)
    #     elif "200K" in str(path):
    #         paths_200k.append(path)
    #     else:
    #         raise ValueError(path)
 
    # print("Number of paths containing 500K points: ", len(paths_200k))
    # print("Number of paths containing 200K points: ", len(paths_500k))

    # move specific files to the destination
    destinations = []
    for path in paths_:
        destination = Path(*path.parts[:4]) / "volume_500K" / Path(*path.parts[5:])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        # destinations.append(destination)
    print(destinations)

    # top 500 samples from sorted files
    # paths_ = sorted(paths_, key=lambda file: str(file))
    # for i, path in enumerate(paths_):
    #     if i >= 500:
    #         break
    #     print(path)

    # reprocess?
    # data_paths = sorted([file for file in Path("/root/hdd2/ShapeNetCoreV2/volume_6M").glob("**/*.npy")
    #                      if file.is_file() and "6M" in file.stem], key=lambda f: str(f))
    # data_paths = [file for file in data_paths if file.parts[-3] == cat_dict["chair"]]

    # print("reprocess {}".format(data_paths[3056]))

    # # for rendering without textures
    # for path in paths_temp:
    #     with open(str(path), "r") as f:
    #         lines = f.readlines()
    #     modified_lines = []

    #     for line in lines:
    #         if "mtllib" not in line.lower():
    #             modified_lines.append(line)
        
    #     with open(str(path.parent / "no_texture.obj"), "w") as f:
    #         f.writelines(modified_lines)
        
    #     print("processed {}".format(path))
    
    # print(len(paths))
    # print(len(paths_))

    # for path in paths:
    #     # remove
    #     # if "model" in str(path):
    #     #     os.remove(str(path))
        
    #     # rename
    #     path.rename(path.with_name("{}_20".format(path.stem) + path.suffix)) # "**/*redered_20.h5"
