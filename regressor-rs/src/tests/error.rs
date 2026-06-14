//! Tests for `utils::error`.

use crate::utils::error::{RegressorError, RegressorResult};

#[test]
fn display_messages_are_stable() {
    assert_eq!(RegressorError::NotFitted.to_string(), "Model not fitted yet");
    assert_eq!(
        RegressorError::SizeMismatch.to_string(),
        "Size of Features and Target dont match"
    );
    assert_eq!(
        RegressorError::InvalidInput("x".into()).to_string(),
        "Invalid Input"
    );
    assert_eq!(RegressorError::EmptyInput.to_string(), "Empty Input");
    assert_eq!(
        RegressorError::Exception("boom".into()).to_string(),
        "Exception: boom"
    );
}

#[test]
fn boxes_into_result_and_downcasts_back() {
    fn fails() -> RegressorResult<()> {
        Err(Box::new(RegressorError::NotFitted))
    }
    let err = fails().unwrap_err();
    assert!(matches!(
        err.downcast_ref::<RegressorError>(),
        Some(RegressorError::NotFitted)
    ));
}
