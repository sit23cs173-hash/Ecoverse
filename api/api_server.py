"""
OPTIONAL: FastAPI Server for Deforestation Detection
Provides REST API endpoints for model inference and analysis.

Run with: uvicorn api_server:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.vegetation_analysis import VegetationIndexCalculator
from analysis.carbon_impact import CarbonImpactCalculator
from models.deforestation_model import NDVIBasedDetector

# Initialize FastAPI app
app = FastAPI(
    title="Deforestation Detection API",
    description="AI-powered deforestation detection and carbon impact assessment",
    version="1.0.0"
)

# Initialize models and calculators
veg_calculator = VegetationIndexCalculator()
carbon_calculator = CarbonImpactCalculator()
ndvi_detector = NDVIBasedDetector()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str


class NDVIRequest(BaseModel):
    image_base64: str  # Base64 encoded image


class CarbonImpactResponse(BaseModel):
    deforested_area_ha: float
    deforested_area_km2: float
    carbon_loss_tons: float
    co2_emissions_tons: float
    equivalent_cars_year: float


class DetectionResponse(BaseModel):
    detection_performed: bool
    deforested_pixels: int
    total_pixels: int
    deforestation_percentage: float
    carbon_impact: CarbonImpactResponse


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Deforestation Detection API is running",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        version="1.0.0"
    )


@app.post("/api/v1/calculate-ndvi")
async def calculate_ndvi(file: UploadFile = File(...)):
    """
    Calculate NDVI from uploaded satellite image.
    
    Args:
        file: Uploaded image file (PNG, JPG, TIFF)
        
    Returns:
        JSON with NDVI statistics
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate NDVI
        ndvi = veg_calculator.calculate_ndvi(image)
        
        # Statistics
        stats = {
            "mean_ndvi": float(ndvi.mean()),
            "median_ndvi": float(np.median(ndvi)),
            "min_ndvi": float(ndvi.min()),
            "max_ndvi": float(ndvi.max()),
            "std_ndvi": float(ndvi.std()),
            "forest_percentage": float((ndvi >= 0.4).sum() / ndvi.size * 100)
        }
        
        return JSONResponse(content=stats)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/detect-deforestation", response_model=DetectionResponse)
async def detect_deforestation(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...)
):
    """
    Detect deforestation from before/after images.
    
    Args:
        before_image: Before satellite image
        after_image: After satellite image
        
    Returns:
        Detection results and carbon impact
    """
    try:
        # Read before image
        contents_before = await before_image.read()
        nparr_before = np.frombuffer(contents_before, np.uint8)
        img_before = cv2.imdecode(nparr_before, cv2.IMREAD_COLOR)
        img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        
        # Read after image
        contents_after = await after_image.read()
        nparr_after = np.frombuffer(contents_after, np.uint8)
        img_after = cv2.imdecode(nparr_after, cv2.IMREAD_COLOR)
        img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
        
        # Calculate NDVI
        ndvi_before = veg_calculator.calculate_ndvi(img_before)
        ndvi_after = veg_calculator.calculate_ndvi(img_after)
        
        # Detect deforestation
        deforestation_mask = ndvi_detector.detect_deforestation(ndvi_before, ndvi_after)
        
        # Calculate carbon impact
        carbon_impact = carbon_calculator.calculate_carbon_from_mask(deforestation_mask)
        
        # Prepare response
        deforested_pixels = int(np.sum(deforestation_mask))
        total_pixels = int(deforestation_mask.size)
        
        response = DetectionResponse(
            detection_performed=True,
            deforested_pixels=deforested_pixels,
            total_pixels=total_pixels,
            deforestation_percentage=float(deforested_pixels / total_pixels * 100),
            carbon_impact=CarbonImpactResponse(**carbon_impact)
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during detection: {str(e)}")


@app.post("/api/v1/estimate-carbon-impact", response_model=CarbonImpactResponse)
async def estimate_carbon_impact(
    deforested_area_ha: float,
    carbon_density: Optional[float] = 190.0,
    forest_type: Optional[str] = "amazon"
):
    """
    Estimate carbon impact from deforested area.
    
    Args:
        deforested_area_ha: Deforested area in hectares
        carbon_density: Carbon density in tons/ha (optional)
        forest_type: Type of forest (optional)
        
    Returns:
        Carbon impact metrics
    """
    try:
        carbon_calc = CarbonImpactCalculator(carbon_density=carbon_density)
        impact = carbon_calc.calculate_carbon_loss(deforested_area_ha, forest_type)
        
        return CarbonImpactResponse(**impact)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating carbon impact: {str(e)}")


@app.get("/api/v1/forest-carbon-densities")
async def get_carbon_densities():
    """
    Get reference carbon density values for different forest types.
    
    Returns:
        Dictionary of forest types and their carbon densities
    """
    densities = {
        "tropical": {
            "carbon_density_tons_per_ha": 180.0,
            "description": "Tropical rainforests"
        },
        "temperate": {
            "carbon_density_tons_per_ha": 90.0,
            "description": "Temperate forests"
        },
        "amazon": {
            "carbon_density_tons_per_ha": 190.0,
            "description": "Amazon rainforest (high biomass)"
        },
        "boreal": {
            "carbon_density_tons_per_ha": 60.0,
            "description": "Boreal/taiga forests"
        }
    }
    
    return JSONResponse(content=densities)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup."""
    print("🚀 Deforestation Detection API starting...")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("✅ All systems ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("🛑 Shutting down API...")


if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("  DEFORESTATION DETECTION API")
    print("="*70)
    print()
    print("Starting FastAPI server...")
    print("Access API documentation at: http://localhost:8000/docs")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
