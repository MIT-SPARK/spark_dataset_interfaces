import click
import rasterio
import rasterio.plot
import pathlib


@click.command()
@click.argument("tiff_path", type=click.Path(exists=True))
@click.argument("trajectories", type=click.Path(exists=True), nargs=-1)
def main(tiff_path, trajectories):
    tiff_path = pathlib.Path(tiff_path).expanduser().absolute()
    trajectories = [pathlib.Path(x).expanduser().absolute() for x in trajectories]

    tiff = rasterio.open(tiff_path)
    print(tiff.bounds)
    rasterio.plot.show(tiff)


if __name__ == "__main__":
    main()
