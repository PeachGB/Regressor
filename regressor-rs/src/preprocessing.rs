use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use serde::{Deserialize, Serialize};

use crate::utils::error::RegressorError;
use crate::utils::error::RegressorResult;

/// Column-wise standardization to zero mean and unit variance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StandardScaler {
    means: Option<Array1<f64>>,
    standard_deviations: Option<Array1<f64>>,
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardScaler {
    pub fn new() -> StandardScaler {
        StandardScaler {
            means: None,
            standard_deviations: None,
        }
    }
    pub fn fit(&mut self, input: &Array2<f64>) -> RegressorResult<()> {
        if input.is_empty() {
            return Err(Box::new(RegressorError::EmptyInput));
        }
        let mean = input
            .columns()
            .into_iter()
            .map(|col| col.mean().unwrap_or(0.0))
            .collect::<Array1<f64>>();
        let std = input
            .columns()
            .into_iter()
            .map(|col| col.std(1.0))
            .collect::<Array1<f64>>();

        self.means = Some(mean);
        self.standard_deviations = Some(std);
        Ok(())
    }
    pub fn transform(&self, input: &Array2<f64>) -> RegressorResult<Array2<f64>> {
        let StandardScaler {
            means: Some(means),
            standard_deviations: Some(standard_deviations),
        } = self
        else {
            return Err(Box::new(RegressorError::NotFitted));
        };
        if input.ncols() != means.len() {
            return Err(Box::new(RegressorError::SizeMismatch));
        }

        // Guard against zero-variance columns: leave them centered (the
        // numerator is already zero) instead of dividing by zero.
        let safe_std = standard_deviations.mapv(|s| if s == 0.0 { 1.0 } else { s });
        let transformed: Array2<f64> = (input.to_owned() - means.view().insert_axis(Axis(0)))
            / safe_std.view().insert_axis(Axis(0));
        Ok(transformed)
    }

    pub fn fit_transform(&mut self, input: &Array2<f64>) -> RegressorResult<Array2<f64>> {
        self.fit(input)?;
        self.transform(input)
    }
}

/// Column-wise rescaling of each feature into the `[0, 1]` range.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MinMaxScaler {
    mins: Option<Array1<f64>>,
    maxes: Option<Array1<f64>>,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxScaler {
    pub fn new() -> MinMaxScaler {
        MinMaxScaler {
            mins: None,
            maxes: None,
        }
    }
    pub fn fit(&mut self, input: &Array2<f64>) -> RegressorResult<()> {
        if input.is_empty() {
            return Err(Box::new(RegressorError::EmptyInput));
        }
        let mins = input
            .columns()
            .into_iter()
            .map(|col| col.iter().copied().fold(f64::INFINITY, f64::min))
            .collect::<Array1<f64>>();
        let maxes = input
            .columns()
            .into_iter()
            .map(|col| col.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect::<Array1<f64>>();
        self.mins = Some(mins);
        self.maxes = Some(maxes);
        Ok(())
    }
    pub fn transform(&self, input: &Array2<f64>) -> RegressorResult<Array2<f64>> {
        let MinMaxScaler {
            mins: Some(mins),
            maxes: Some(maxes),
        } = self
        else {
            return Err(Box::new(RegressorError::NotFitted));
        };
        if input.ncols() != mins.len() {
            return Err(Box::new(RegressorError::SizeMismatch));
        }
        // Zero-range columns map to 0.0 rather than dividing by zero.
        let range = (maxes - mins).mapv(|r| if r == 0.0 { 1.0 } else { r });
        let transformed: Array2<f64> =
            (input.to_owned() - mins.view().insert_axis(Axis(0))) / range.view().insert_axis(Axis(0));
        Ok(transformed)
    }
    pub fn fit_transform(&mut self, input: &Array2<f64>) -> RegressorResult<Array2<f64>> {
        self.fit(input)?;
        self.transform(input)
    }
}

/// Map categorical labels to consecutive integer codes (`0..n_classes`).
///
/// Classes are ordered lexicographically, mirroring scikit-learn's
/// `LabelEncoder`. Codes are returned as `f64` so they slot straight into the
/// `ndarray`-based numeric pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelEncoder {
    classes: Option<Vec<String>>,
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelEncoder {
    pub fn new() -> LabelEncoder {
        LabelEncoder { classes: None }
    }

    /// Learn the sorted set of unique labels.
    pub fn fit(&mut self, labels: &[String]) -> RegressorResult<()> {
        if labels.is_empty() {
            return Err(Box::new(RegressorError::EmptyInput));
        }
        let mut classes: Vec<String> = labels.to_vec();
        classes.sort();
        classes.dedup();
        self.classes = Some(classes);
        Ok(())
    }

    /// Encode `labels` into their integer codes.
    pub fn transform(&self, labels: &[String]) -> RegressorResult<Array1<f64>> {
        let Some(classes) = &self.classes else {
            return Err(Box::new(RegressorError::NotFitted));
        };
        let mut codes = Vec::with_capacity(labels.len());
        for label in labels {
            match classes.iter().position(|c| c == label) {
                Some(idx) => codes.push(idx as f64),
                None => {
                    return Err(Box::new(RegressorError::InvalidInput(format!(
                        "unknown label: {label}"
                    ))));
                }
            }
        }
        Ok(Array1::from_vec(codes))
    }

    pub fn fit_transform(&mut self, labels: &[String]) -> RegressorResult<Array1<f64>> {
        self.fit(labels)?;
        self.transform(labels)
    }

    /// Map integer codes back to their original labels.
    pub fn inverse_transform(&self, codes: &[f64]) -> RegressorResult<Vec<String>> {
        let Some(classes) = &self.classes else {
            return Err(Box::new(RegressorError::NotFitted));
        };
        let mut labels = Vec::with_capacity(codes.len());
        for &code in codes {
            let idx = code as usize;
            match classes.get(idx) {
                Some(label) if code >= 0.0 && code.fract() == 0.0 => labels.push(label.clone()),
                _ => {
                    return Err(Box::new(RegressorError::InvalidInput(format!(
                        "code out of range: {code}"
                    ))));
                }
            }
        }
        Ok(labels)
    }

    /// The learned classes, in code order.
    pub fn classes(&self) -> Option<&[String]> {
        self.classes.as_deref()
    }
}
