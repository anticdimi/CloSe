## Setup Interactive Tool (Optional)

To setup the interactive tool, after setting up the conda environment, run:

```
cd Open3D-CloSeT
bash make.sh
```

**Notes**

Compilation and installation of the interactive tool might take a while. Refer to [Open3D-CloSeT](https://github.com/Bozcomlekci/Open3D-CloSeT/tree/f97b3f0debbc8a120eefd04706889d0c2dbe36ba) repository for the details.

## Running the Tool

To work with the CloSe-T Interactive Tool, use the provided .vscode launch configuration or run:

```
python interactive_tool.py  --pretrained_path ./pretrained/closenet.pth
```

### Inference and Evaluation using the Tool

To perform inference and evalution using the tool, make sure that your data follows the specifications and steps described in [dataset.md](./dataset.md).
