mod cli;
mod colormap;
mod discovery;
mod fulfillment;
mod isosurface;
mod mesh;
mod viewer;

use crate::viewer::AmrexViewerPlugin;

fn main() {
    tephrite_rs::run(AmrexViewerPlugin);
}
