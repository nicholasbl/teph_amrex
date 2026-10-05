mod cli;
mod colormap;
mod convert;
mod discovery;
mod fulfillment;
mod isosurface;
mod mesh;
mod viewer;

use anyhow::Result;
use clap::Parser;

use crate::{cli::Args, convert::convert_directory, viewer::AmrexViewerPlugin};

fn main() -> Result<()> {
    let Ok(args) = Args::try_parse() else {
        tephrite_rs::run(AmrexViewerPlugin { dir: None });
        return Ok(());
    };

    if args.convert {
        convert_directory(&args.dir, &args.quantities)
    } else {
        let dir = args.dir.clone();
        tephrite_rs::run(AmrexViewerPlugin { dir: Some(dir) });
        Ok(())
    }
}
