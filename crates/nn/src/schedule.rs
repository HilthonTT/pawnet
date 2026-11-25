use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LrSchedule {
    /// The learning rate never changes.
    Constant,
    /// `lr = base · decayᵉᵖᵒᶜʰ`
    Exponential { decay: f32 },
    /// `lr = base · factor^⌊epoch / every⌋`: drops by `factor` every `every` epochs.
    Step { factor: f32, every: usize },
    /// `lr = base / (1 + decay · epoch)`
    TimeBased { decay: f32 },
    /// Half a cosine wave from `base` down to `base · min` over the run.
    Cosine { min: f32 },
}

impl Default for LrSchedule {
    fn default() -> Self {
        LrSchedule::Exponential { decay: 0.9 }
    }
}

impl LrSchedule {
    /// Checks that the schedule's parameters keep the learning rate finite and positive.
    pub fn check(&self) -> Result<()> {
        let positive = |v: f32| v.is_finite() && v > 0.0;
        let valid = match *self {
            LrSchedule::Constant => true,
            LrSchedule::Exponential { decay } => positive(decay),
            LrSchedule::Step { factor, every } => positive(factor) && every > 0,
            LrSchedule::TimeBased { decay } => decay.is_finite() && decay >= 0.0,
            LrSchedule::Cosine { min } => (0.0..=1.0).contains(&min),
        };
        if valid {
            Ok(())
        } else {
            Err(Error::invalid_parameter(format!(
                "invalid schedule `{self}`: decay factors must be positive, the step interval \
                 at least 1 and the cosine floor in [0, 1]"
            )))
        }
    }

    pub fn rate(&self, base: f32, epoch: usize, epochs: usize) -> f32 {
        match *self {
            LrSchedule::Constant => base,
            LrSchedule::Exponential { decay } => base * decay.powi(epoch as i32),
            LrSchedule::Step { factor, every } => base * factor.powi((epoch / every.max(1)) as i32),
            LrSchedule::TimeBased { decay } => base / (1.0 + decay * epoch as f32),
            LrSchedule::Cosine { min } => {
                let progress = epoch as f32 / epochs.max(1) as f32;
                let floor = base * min;
                floor + (base - floor) * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
            }
        }
    }
}

/// Formats as the same syntax [`FromStr`] accepts, e.g. `exp:0.9` or `step:0.5:5`.
impl fmt::Display for LrSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LrSchedule::Constant => write!(f, "constant"),
            LrSchedule::Exponential { decay } => write!(f, "exp:{decay}"),
            LrSchedule::Step { factor, every } => write!(f, "step:{factor}:{every}"),
            LrSchedule::TimeBased { decay } => write!(f, "time:{decay}"),
            LrSchedule::Cosine { min } if *min == 0.0 => write!(f, "cosine"),
            LrSchedule::Cosine { min } => write!(f, "cosine:{min}"),
        }
    }
}

impl FromStr for LrSchedule {
    type Err = String;

    /// Parses `constant`, `exp:<decay>`, `step:<factor>:<every>`, `time:<decay>`, `cosine` or
    /// `cosine:<min>`, rejecting values that [`LrSchedule::check`] would.
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        let err = || {
            format!(
                "invalid schedule `{s}` (expected constant, exp:0.9, step:0.5:5, time:0.1 or cosine)"
            )
        };
        let parts: Vec<&str> = s.split(':').collect();
        let num = |i: usize| {
            parts
                .get(i)
                .and_then(|p| p.parse::<f32>().ok())
                .ok_or_else(err)
        };
        let schedule = match parts[0] {
            "constant" if parts.len() == 1 => LrSchedule::Constant,
            "exp" if parts.len() == 2 => LrSchedule::Exponential { decay: num(1)? },
            "time" if parts.len() == 2 => LrSchedule::TimeBased { decay: num(1)? },
            "cosine" if parts.len() == 1 => LrSchedule::Cosine { min: 0.0 },
            "cosine" if parts.len() == 2 => LrSchedule::Cosine { min: num(1)? },
            "step" if parts.len() == 3 => {
                let every = parts[2].parse().map_err(|_| err())?;
                LrSchedule::Step {
                    factor: num(1)?,
                    every,
                }
            }
            _ => return Err(err()),
        };
        schedule.check().map_err(|e| e.to_string())?;
        Ok(schedule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_rates() {
        assert_eq!(LrSchedule::Constant.rate(0.1, 7, 10), 0.1);
        assert!((LrSchedule::Exponential { decay: 0.5 }.rate(1.0, 3, 10) - 0.125).abs() < 1e-6);
        let step = LrSchedule::Step {
            factor: 0.1,
            every: 5,
        };
        assert_eq!(step.rate(1.0, 4, 10), 1.0);
        assert!((step.rate(1.0, 5, 10) - 0.1).abs() < 1e-6);
        assert_eq!(LrSchedule::TimeBased { decay: 1.0 }.rate(1.0, 3, 10), 0.25);
        let cosine = LrSchedule::Cosine { min: 0.1 };
        assert_eq!(cosine.rate(2.0, 0, 10), 2.0);
        assert!((cosine.rate(2.0, 5, 10) - 1.1).abs() < 1e-6);
        assert!((cosine.rate(2.0, 10, 10) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn parses_and_displays() {
        for text in [
            "constant",
            "exp:0.9",
            "step:0.5:5",
            "time:0.1",
            "cosine",
            "cosine:0.01",
        ] {
            let schedule: LrSchedule = text.parse().unwrap();
            assert_eq!(schedule.to_string(), text);
        }
        assert!("step:0.5".parse::<LrSchedule>().is_err());
        assert!("exp:abc".parse::<LrSchedule>().is_err());
    }

    #[test]
    fn rejects_schedules_that_break_the_learning_rate() {
        for text in [
            "exp:-1",
            "exp:0",
            "exp:nan",
            "step:0.5:0",
            "step:-0.5:2",
            "time:-1",
            "time:inf",
            "cosine:2",
            "cosine:-0.1",
        ] {
            assert!(text.parse::<LrSchedule>().is_err(), "{text} was accepted");
        }
        assert!(
            LrSchedule::Step {
                factor: 0.5,
                every: 0
            }
            .check()
            .is_err()
        );
        assert!(LrSchedule::default().check().is_ok());
    }
}
