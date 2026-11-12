use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(version, about)]
pub(crate) struct Args {
    /// Directory containing MSG archive timesteps.
    pub(crate) dir: PathBuf,
}
