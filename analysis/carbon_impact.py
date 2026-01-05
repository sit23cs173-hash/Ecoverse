"""
STEP 7: CARBON IMPACT ASSESSMENT MODULE
Estimates carbon stock loss from deforestation using forest carbon density values.
Calculates CO2 emissions and provides carbon impact reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarbonImpactCalculator:
    """
    Calculate carbon emissions and carbon stock loss from deforestation.
    """
    
    def __init__(self,
                 carbon_density: float = 190.0,
                 pixel_area_ha: float = 0.01,
                 co2_conversion_factor: float = 3.67):
        """
        Initialize carbon impact calculator.
        
        Args:
            carbon_density: Average carbon density in tons C/hectare
                          (Default: 190 for Amazon rainforest)
            pixel_area_ha: Area represented by one pixel in hectares
                          (Default: 0.01 for Sentinel-2 at 10m resolution)
            co2_conversion_factor: Factor to convert C to CO2 (1 ton C = 3.67 tons CO2)
        """
        self.carbon_density = carbon_density
        self.pixel_area_ha = pixel_area_ha
        self.co2_conversion_factor = co2_conversion_factor
        
        logger.info(f"Initialized CarbonImpactCalculator:")
        logger.info(f"  Carbon density: {carbon_density} tons C/ha")
        logger.info(f"  Pixel area: {pixel_area_ha} ha")
        logger.info(f"  CO2 conversion: {co2_conversion_factor}")
    
    def calculate_deforested_area(self, 
                                 deforestation_mask: np.ndarray) -> float:
        """
        Calculate total deforested area from binary mask.
        
        Args:
            deforestation_mask: Binary mask where 1 = deforested
            
        Returns:
            Deforested area in hectares
        """
        # Count deforested pixels
        deforested_pixels = np.sum(deforestation_mask > 0)
        
        # Convert to hectares
        area_ha = deforested_pixels * self.pixel_area_ha
        
        logger.info(f"Deforested area: {area_ha:.2f} hectares ({deforested_pixels:,} pixels)")
        
        return area_ha
    
    def calculate_carbon_loss(self, 
                             deforested_area_ha: float,
                             forest_type: str = 'tropical') -> Dict[str, float]:
        """
        Calculate carbon loss from deforestation.
        
        Args:
            deforested_area_ha: Deforested area in hectares
            forest_type: Type of forest ('tropical', 'temperate', 'amazon')
            
        Returns:
            Dictionary with carbon loss metrics
        """
        # Adjust carbon density based on forest type if needed
        if forest_type == 'tropical':
            carbon_density = 180.0
        elif forest_type == 'temperate':
            carbon_density = 90.0
        elif forest_type == 'amazon':
            carbon_density = self.carbon_density
        else:
            carbon_density = self.carbon_density
        
        # Calculate carbon stock loss (in tons of carbon)
        carbon_loss_tons = deforested_area_ha * carbon_density
        
        # Convert to CO2 emissions
        co2_emissions_tons = carbon_loss_tons * self.co2_conversion_factor
        
        # Additional metrics
        results = {
            'deforested_area_ha': deforested_area_ha,
            'deforested_area_km2': deforested_area_ha / 100,
            'carbon_density_used': carbon_density,
            'carbon_loss_tons': carbon_loss_tons,
            'co2_emissions_tons': co2_emissions_tons,
            'co2_emissions_kg': co2_emissions_tons * 1000,
            'equivalent_cars_year': co2_emissions_tons / 4.6  # Average car emits ~4.6 tons CO2/year
        }
        
        logger.info(f"Carbon Impact:")
        logger.info(f"  Area: {deforested_area_ha:.2f} ha ({results['deforested_area_km2']:.2f} km²)")
        logger.info(f"  Carbon loss: {carbon_loss_tons:.2f} tons C")
        logger.info(f"  CO2 emissions: {co2_emissions_tons:.2f} tons CO2")
        logger.info(f"  Equivalent to {results['equivalent_cars_year']:.0f} cars for one year")
        
        return results
    
    def calculate_carbon_from_mask(self, 
                                  deforestation_mask: np.ndarray,
                                  forest_type: str = 'amazon') -> Dict[str, float]:
        """
        Calculate carbon impact directly from deforestation mask.
        
        Args:
            deforestation_mask: Binary mask where 1 = deforested
            forest_type: Type of forest
            
        Returns:
            Dictionary with carbon impact metrics
        """
        # Calculate deforested area
        area_ha = self.calculate_deforested_area(deforestation_mask)
        
        # Calculate carbon loss
        carbon_impact = self.calculate_carbon_loss(area_ha, forest_type)
        
        return carbon_impact
    
    def calculate_regional_carbon_loss(self,
                                      masks_by_region: Dict[str, np.ndarray],
                                      forest_type: str = 'amazon') -> pd.DataFrame:
        """
        Calculate carbon loss for multiple regions.
        
        Args:
            masks_by_region: Dictionary mapping region names to deforestation masks
            forest_type: Type of forest
            
        Returns:
            DataFrame with carbon loss by region
        """
        logger.info(f"Calculating carbon loss for {len(masks_by_region)} regions")
        
        results = []
        
        for region_name, mask in masks_by_region.items():
            impact = self.calculate_carbon_from_mask(mask, forest_type)
            impact['region'] = region_name
            results.append(impact)
        
        df = pd.DataFrame(results)
        df = df[['region', 'deforested_area_ha', 'deforested_area_km2', 
                'carbon_loss_tons', 'co2_emissions_tons', 'equivalent_cars_year']]
        
        return df
    
    def calculate_temporal_carbon_loss(self,
                                      masks_by_time: Dict[str, np.ndarray],
                                      forest_type: str = 'amazon') -> pd.DataFrame:
        """
        Calculate carbon loss over time periods.
        
        Args:
            masks_by_time: Dictionary mapping time periods to deforestation masks
            forest_type: Type of forest
            
        Returns:
            DataFrame with temporal carbon loss
        """
        logger.info(f"Calculating temporal carbon loss for {len(masks_by_time)} time periods")
        
        results = []
        
        for time_period, mask in sorted(masks_by_time.items()):
            impact = self.calculate_carbon_from_mask(mask, forest_type)
            impact['time_period'] = time_period
            results.append(impact)
        
        df = pd.DataFrame(results)
        df = df[['time_period', 'deforested_area_ha', 'carbon_loss_tons', 
                'co2_emissions_tons', 'equivalent_cars_year']]
        
        # Calculate cumulative emissions
        df['cumulative_co2_tons'] = df['co2_emissions_tons'].cumsum()
        
        return df
    
    def estimate_economic_value(self,
                               co2_emissions_tons: float,
                               carbon_price_per_ton: float = 50.0) -> Dict[str, float]:
        """
        Estimate economic value of carbon loss using carbon pricing.
        
        Args:
            co2_emissions_tons: CO2 emissions in tons
            carbon_price_per_ton: Price per ton of CO2 (default: $50/ton)
            
        Returns:
            Dictionary with economic estimates
        """
        economic_loss = co2_emissions_tons * carbon_price_per_ton
        
        results = {
            'co2_emissions_tons': co2_emissions_tons,
            'carbon_price_per_ton_usd': carbon_price_per_ton,
            'economic_loss_usd': economic_loss,
            'economic_loss_million_usd': economic_loss / 1_000_000
        }
        
        logger.info(f"Economic Impact:")
        logger.info(f"  Carbon price: ${carbon_price_per_ton}/ton CO2")
        logger.info(f"  Economic loss: ${economic_loss:,.2f}")
        
        return results


class CarbonImpactVisualizer:
    """
    Visualize carbon impact assessments.
    """
    
    @staticmethod
    def plot_carbon_loss_by_region(df: pd.DataFrame, 
                                   save_path: Optional[str] = None):
        """
        Plot carbon loss by region.
        
        Args:
            df: DataFrame with regional carbon loss data
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Deforested area by region
        axes[0].barh(df['region'], df['deforested_area_ha'], color='green', alpha=0.7)
        axes[0].set_xlabel('Deforested Area (hectares)', fontsize=11)
        axes[0].set_title('Deforested Area by Region', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # CO2 emissions by region
        axes[1].barh(df['region'], df['co2_emissions_tons'], color='red', alpha=0.7)
        axes[1].set_xlabel('CO2 Emissions (tons)', fontsize=11)
        axes[1].set_title('Carbon Emissions by Region', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved regional carbon loss plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_temporal_carbon_loss(df: pd.DataFrame,
                                  save_path: Optional[str] = None):
        """
        Plot carbon loss over time.
        
        Args:
            df: DataFrame with temporal carbon loss data
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Annual emissions
        axes[0].plot(df['time_period'], df['co2_emissions_tons'],
                    marker='o', color='red', linewidth=2, markersize=8)
        axes[0].set_xlabel('Time Period', fontsize=11)
        axes[0].set_ylabel('CO2 Emissions (tons)', fontsize=11)
        axes[0].set_title('Annual Carbon Emissions from Deforestation', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Cumulative emissions
        axes[1].fill_between(df['time_period'], df['cumulative_co2_tons'],
                            color='darkred', alpha=0.6)
        axes[1].plot(df['time_period'], df['cumulative_co2_tons'],
                    color='darkred', linewidth=2)
        axes[1].set_xlabel('Time Period', fontsize=11)
        axes[1].set_ylabel('Cumulative CO2 Emissions (tons)', fontsize=11)
        axes[1].set_title('Cumulative Carbon Emissions', 
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved temporal carbon loss plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_carbon_impact_summary(impact_dict: Dict[str, float],
                                  save_path: Optional[str] = None):
        """
        Create summary visualization of carbon impact.
        
        Args:
            impact_dict: Dictionary with carbon impact metrics
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Deforested area
        axes[0, 0].bar(['Area'], [impact_dict['deforested_area_ha']], 
                      color='brown', alpha=0.7)
        axes[0, 0].set_ylabel('Hectares')
        axes[0, 0].set_title('Deforested Area', fontweight='bold')
        axes[0, 0].text(0, impact_dict['deforested_area_ha'] * 0.5,
                       f"{impact_dict['deforested_area_ha']:.1f} ha",
                       ha='center', fontsize=12, fontweight='bold')
        
        # Carbon loss
        axes[0, 1].bar(['Carbon Loss'], [impact_dict['carbon_loss_tons']],
                      color='gray', alpha=0.7)
        axes[0, 1].set_ylabel('Tons of Carbon')
        axes[0, 1].set_title('Carbon Stock Loss', fontweight='bold')
        axes[0, 1].text(0, impact_dict['carbon_loss_tons'] * 0.5,
                       f"{impact_dict['carbon_loss_tons']:.1f} tons",
                       ha='center', fontsize=12, fontweight='bold')
        
        # CO2 emissions
        axes[1, 0].bar(['CO2 Emissions'], [impact_dict['co2_emissions_tons']],
                      color='red', alpha=0.7)
        axes[1, 0].set_ylabel('Tons of CO2')
        axes[1, 0].set_title('CO2 Emissions', fontweight='bold')
        axes[1, 0].text(0, impact_dict['co2_emissions_tons'] * 0.5,
                       f"{impact_dict['co2_emissions_tons']:.1f} tons",
                       ha='center', fontsize=12, fontweight='bold')
        
        # Equivalent cars
        axes[1, 1].bar(['Car Equivalent'], [impact_dict['equivalent_cars_year']],
                      color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('Number of Cars')
        axes[1, 1].set_title('Equivalent to Cars per Year', fontweight='bold')
        axes[1, 1].text(0, impact_dict['equivalent_cars_year'] * 0.5,
                       f"{impact_dict['equivalent_cars_year']:.0f} cars",
                       ha='center', fontsize=12, fontweight='bold')
        
        plt.suptitle('Carbon Impact Summary', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved carbon impact summary to {save_path}")
        
        plt.show()


def generate_carbon_report(impact_dict: Dict[str, float],
                          output_path: Optional[str] = None) -> str:
    """
    Generate a text report of carbon impact.
    
    Args:
        impact_dict: Dictionary with carbon impact metrics
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    report = f"""
{'='*70}
               CARBON IMPACT ASSESSMENT REPORT
{'='*70}

DEFORESTATION AREA:
  - Area deforested: {impact_dict['deforested_area_ha']:,.2f} hectares
  - Area deforested: {impact_dict['deforested_area_km2']:,.2f} square kilometers

CARBON STOCK LOSS:
  - Carbon density: {impact_dict['carbon_density_used']:.1f} tons C/hectare
  - Total carbon loss: {impact_dict['carbon_loss_tons']:,.2f} tons of carbon

CO2 EMISSIONS:
  - Total CO2 emissions: {impact_dict['co2_emissions_tons']:,.2f} tons of CO2
  - Total CO2 emissions: {impact_dict['co2_emissions_kg']:,.2f} kg of CO2

EQUIVALENCE:
  - Equivalent to emissions from {impact_dict['equivalent_cars_year']:,.0f} cars 
    driven for one year

CONTEXT:
  - 1 ton of carbon = {3.67:.2f} tons of CO2
  - Average passenger car emits ~4.6 tons CO2 per year
  - Forests are critical carbon sinks that regulate climate

{'='*70}
"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Carbon report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    print("Carbon Impact Assessment Module")
    print("=" * 70)
    
    # Create dummy deforestation mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 1  # 100x100 deforested region
    
    # Initialize calculator
    calculator = CarbonImpactCalculator(
        carbon_density=190.0,
        pixel_area_ha=0.01
    )
    
    # Calculate impact
    impact = calculator.calculate_carbon_from_mask(mask, forest_type='amazon')
    
    print("\nCarbon Impact Results:")
    for key, value in impact.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Generate report
    report = generate_carbon_report(impact)
    print(report)
    
    # Visualize
    visualizer = CarbonImpactVisualizer()
    visualizer.plot_carbon_impact_summary(impact)
