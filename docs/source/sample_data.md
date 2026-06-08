# Obtaining sample data

Running the turbocluster examples requires an Arepo snapshot of a cosmological simulation. A suitable snapshot can be found in the `data` folder of the [Paicos repository](https://github.com/tberlok/paicos) and can be downloaded on your local system. To do so, follow the following steps:

1. Create a directory on your filesystem, e.g. `mkdir /path/to/sample_data` and `cd` there
2. Type in the following commands to download the snapshot (or dowload them directly from GitHub):
```
wget -O snap_247.hdf5 https://sid.erda.dk/share_redirect/G4pUGFJUpq
wget -O fof_subhalo_tab_247.hdf5 https://sid.erda.dk/share_redirect/BmILDaDPPz
```
3. Navigate to the paicos root directory. If you installed turbocluster in a conda environment, and followed the steps outlined above, then the paicos directory will be in the `site-packages` directory of that conda environment. One simple way to find it is as follows:
    - Activate the conda environment created during the installation process, e.g. `conda activate turbocluster`
    - Type: `import paicos as pa`
    - The root directory is: `print(pa.root_dir)` 
4. Once you are in the paicos root directory, create a copy of the `paicos_user_settings_template.py` as:
```
cp paicos_user_settings_template.py paicos_user_settings.py
```
5. Use your text editor of choice and replace the string `data_dir = '/.../'` in `paicos_user_settings.py` with the path to your snapshot folder, so that it reads, e.g., `data_dir = '/path/to/sample_data/'`
6. Open a new terminal in your conda environment, import paicos, and double checked that `pa.root_dir` is now the same as your snapshot directory. If so, the snapshot can now be loaded as
```
snap = pa.Snapshot(pa.data_dir, 247)
```