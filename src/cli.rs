use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(version, about)]
pub(crate) struct Args {
    /// Directory containing AMReX plotfiles, .packed files, or .compact files.
    pub(crate) dir: PathBuf,

    /// Convert plotfiles to .packed files and exit.
    #[arg(long)]
    pub(crate) convert: bool,

    /// Quantities to include when converting. Empty means all quantities.
    #[arg(short, long, value_name = "QUANTITY")]
    pub(crate) quantities: Vec<String>,
}
