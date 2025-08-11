from pydantic import BaseModel, Field, field_validator

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")
    
    @field_validator('*')
    @classmethod
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError('Features must be positive')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    features: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    
    model_config = {
        'protected_namespaces': ()
    }