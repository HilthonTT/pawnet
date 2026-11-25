use std::fmt;
use std::io;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Io(io::Error),
    /// A file (model, dataset or image) is malformed.
    Format(String),
    /// A hyperparameter is out of range, e.g. a negative learning rate.
    InvalidParameter(String),
}

impl Error {
    pub(crate) fn format(message: impl Into<String>) -> Self {
        Error::Format(message.into())
    }

    pub(crate) fn invalid_parameter(message: impl Into<String>) -> Self {
        Error::InvalidParameter(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Format(message) | Error::InvalidParameter(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Format(_) | Error::InvalidParameter(_) => None,
        }
    }
}

impl From<pawnet_nn::Error> for Error {
    fn from(e: pawnet_nn::Error) -> Self {
        match e {
            pawnet_nn::Error::Io(e) => Error::Io(e),
            pawnet_nn::Error::InvalidParameter(message) => Error::InvalidParameter(message),
            pawnet_nn::Error::Format(message) => Error::Format(message),
            other => Error::Format(other.to_string()),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
