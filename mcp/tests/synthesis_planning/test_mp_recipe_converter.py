"""
Tests for MP recipe converter tool.

This module tests the convert_mp_recipes_to_synthesis_routes tool which converts
Materials Project synthesis recipes into MatClaw's standardized synthesis route format.
"""

import pytest
from tools.synthesis_planning import convert_mp_recipes_to_synthesis_routes


class TestMPRecipeConverter:
    """Test convert_mp_recipes_to_synthesis_routes function."""
    
    def test_basic_recipe_conversion(self):
        """Test conversion of a basic Materials Project recipe."""
        mp_recipes = [
            {
                "recipe_id": "mp-19017_1",
                "target_formula": "LiFePO4",
                "target_material_id": "mp-19017",
                "precursors": [
                    {"formula": "Li2CO3", "amount": 0.5, "form": "carbonate"},
                    {"formula": "Fe2O3", "amount": 0.5, "form": "oxide"},
                    {"formula": "NH4H2PO4", "amount": 1.0, "form": "phosphate"}
                ],
                "operations": "Mix precursors thoroughly. Heat at 600°C for 8 hours.",
                "temperature_celsius": 600,
                "heating_time_hours": 8,
                "atmosphere": "air",
                "citation": "Smith et al., Journal of Materials, 2020",
                "doi": "10.1234/jmat.2020.001",
                "year": 2020
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="LiFePO4",
            constraints={}
        )
        
        assert result["success"] is True
        assert result["target_composition"] == "LiFePO4"
        assert result["n_routes"] == 1
        
        route = result["routes"][0]
        assert route["source"] == "literature"
        assert route["confidence"] == 0.90
        assert "precursors" in route
        assert len(route["precursors"]) == 3
        assert "steps" in route
        assert route["citation"] == "Smith et al., Journal of Materials, 2020"
        assert route["doi"] == "10.1234/jmat.2020.001"
    
    def test_empty_recipe_list(self):
        """Test handling of empty recipe list."""
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=[],
            target_composition="LiCoO2",
            constraints={}
        )
        
        assert result["success"] is True
        assert result["n_routes"] == 0
        assert len(result["routes"]) == 0
        assert result["warnings"] is not None
        assert "No recipes" in result["warnings"][0]
    
    def test_temperature_filtering(self):
        """Test that recipes exceeding max temperature are filtered out."""
        mp_recipes = [
            {
                "recipe_id": "mp-001_1",
                "target_formula": "LiCoO2",
                "precursors": [{"formula": "Li2CO3"}, {"formula": "Co3O4"}],
                "operations": "Heat at 900°C",
                "temperature_celsius": 900,
                "heating_time_hours": 10
            },
            {
                "recipe_id": "mp-001_2",
                "target_formula": "LiCoO2",
                "precursors": [{"formula": "LiOH"}, {"formula": "CoO"}],
                "operations": "Heat at 700°C",
                "temperature_celsius": 700,
                "heating_time_hours": 8
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="LiCoO2",
            constraints={"max_temperature": 800}
        )
        
        assert result["success"] is True
        assert result["n_routes"] == 1  # Only the 700°C recipe should pass
        assert result["routes"][0]["temperature_range"] == "700°C"
    
    def test_time_filtering(self):
        """Test that recipes exceeding max time are filtered out."""
        mp_recipes = [
            {
                "recipe_id": "mp-002_1",
                "target_formula": "NiO",
                "precursors": [{"formula": "Ni(NO3)2"}],
                "operations": "Heat at 800°C for 15 hours",
                "temperature_celsius": 800,
                "heating_time_hours": 15
            },
            {
                "recipe_id": "mp-002_2",
                "target_formula": "NiO",
                "precursors": [{"formula": "NiCO3"}],
                "operations": "Heat at 800°C for 5 hours",
                "temperature_celsius": 800,
                "heating_time_hours": 5
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="NiO",
            constraints={"max_time": 10}
        )
        
        assert result["success"] is True
        assert result["n_routes"] == 1  # Only the 5-hour recipe should pass
        assert "5" in result["routes"][0]["total_time_estimate"]
    
    def test_method_inference_hydrothermal(self):
        """Test inference of hydrothermal method from recipe conditions."""
        mp_recipes = [
            {
                "recipe_id": "mp-003_1",
                "target_formula": "LiFePO4",
                "precursors": [{"formula": "LiOH"}, {"formula": "FeSO4"}, {"formula": "H3PO4"}],
                "operations": "Place in hydrothermal autoclave at 180°C for 12 hours",
                "temperature_celsius": 180,
                "heating_time_hours": 12,
                "conditions": "hydrothermal autoclave"
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="LiFePO4",
            constraints={}
        )
        
        assert result["success"] is True
        route = result["routes"][0]
        assert route["method"] == "hydrothermal"
    
    def test_method_inference_solution(self):
        """Test inference of solution method from recipe conditions."""
        mp_recipes = [
            {
                "recipe_id": "mp-004_1",
                "target_formula": "CaCO3",
                "precursors": [{"formula": "CaCl2"}, {"formula": "Na2CO3"}],
                "operations": "Dissolve in water and precipitate",
                "temperature_celsius": 25,
                "heating_time_hours": 2,
                "conditions": "solution precipitation"
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="CaCO3",
            constraints={}
        )
        
        assert result["success"] is True
        route = result["routes"][0]
        assert route["method"] == "solution"
    
    def test_method_inference_solid_state_default(self):
        """Test that solid_state is the default method."""
        mp_recipes = [
            {
                "recipe_id": "mp-005_1",
                "target_formula": "BaTiO3",
                "precursors": [{"formula": "BaCO3"}, {"formula": "TiO2"}],
                "operations": "Mix and heat at 1200°C for 10 hours",
                "temperature_celsius": 1200,
                "heating_time_hours": 10
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="BaTiO3",
            constraints={}
        )
        
        assert result["success"] is True
        route = result["routes"][0]
        assert route["method"] == "solid_state"
    
    def test_invalid_composition_format(self):
        """Test handling of invalid composition."""
        mp_recipes = [{"recipe_id": "mp-006_1", "target_formula": "LiCoO2", "precursors": [], "operations": "Heat"}]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="InvalidXYZ123",
            constraints={}
        )
        
        # Invalid composition should generate a warning but still process recipes
        assert result["success"] is True
        assert result["warnings"] is not None
    
    def test_missing_composition_field(self):
        """Test handling when composition is empty string."""
        mp_recipes = [{"recipe_id": "mp-007_1", "target_formula": "LiCoO2", "precursors": [], "operations": "Heat"}]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="",  # Empty composition to test error handling
            constraints={}
        )
        
        # Empty composition should generate a warning but still process recipes
        assert result["success"] is True
        assert result["warnings"] is not None
    
    def test_precursor_extraction_with_minimal_data(self):
        """Test precursor extraction when data is minimal."""
        mp_recipes = [
            {
                "recipe_id": "mp-008_1",
                "target_formula": "MgO",
                "precursors": ["Mg(NO3)2"],  # String list instead of dict list
                "operations": "Heat at 800°C",
                "temperature_celsius": 800,
                "heating_time_hours": 6
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="MgO",
            constraints={}
        )
        
        assert result["success"] is True
        route = result["routes"][0]
        assert len(route["precursors"]) == 1
        assert route["precursors"][0]["compound"] == "Mg(NO3)2"
        assert route["precursors"][0]["form"] == "unspecified"
    
    def test_operations_parsing(self):
        """Test parsing of operations string into steps."""
        mp_recipes = [
            {
                "recipe_id": "mp-009_1",
                "target_formula": "ZnO",
                "precursors": [{"formula": "Zn(NO3)2"}],
                "operations": "Grind precursors. Heat at 500°C for 4 hours. Cool to room temperature.",
                "temperature_celsius": 500,
                "heating_time_hours": 4
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="ZnO",
            constraints={}
        )
        
        assert result["success"] is True
        route = result["routes"][0]
        assert "steps" in route
        assert len(route["steps"]) >= 1  # Should have at least one step
        # Check that temperature was extracted
        temp_found = any(
            step.get("temperature_c") == 500 
            for step in route["steps"] 
            if step.get("temperature_c")
        )
        assert temp_found
    
    def test_feasibility_scoring(self):
        """Test that literature routes get appropriate feasibility scores."""
        mp_recipes = [
            {
                "recipe_id": "mp-010_1",
                "target_formula": "LiCoO2",
                "precursors": [{"formula": "Li2CO3"}, {"formula": "Co3O4"}],
                "operations": "Heat at 900°C for 12 hours",
                "temperature_celsius": 900,
                "heating_time_hours": 12
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="LiCoO2",
            constraints={}
        )
        
        assert result["success"] is True
        route = result["routes"][0]
        # Literature routes should have high confidence
        assert route["confidence"] == 0.90
        # Feasibility score should be reasonable
        assert 0.0 <= route["feasibility_score"] <= 1.0
        assert route["feasibility_score"] >= 0.70  # Literature routes should score well
    
    def test_route_ranking_by_feasibility(self):
        """Test that routes are ranked by feasibility score."""
        mp_recipes = [
            {
                "recipe_id": "mp-011_1",
                "target_formula": "NiO",
                "precursors": [{"formula": "Ni(NO3)2"}],
                "operations": "Heat at 1200°C for 20 hours",  # High temp, long time
                "temperature_celsius": 1200,
                "heating_time_hours": 20
            },
            {
                "recipe_id": "mp-011_2",
                "target_formula": "NiO",
                "precursors": [{"formula": "NiCO3"}],
                "operations": "Heat at 600°C for 4 hours",  # Moderate conditions
                "temperature_celsius": 600,
                "heating_time_hours": 4
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="NiO",
            constraints={}
        )
        
        assert result["success"] is True
        assert result["n_routes"] == 2
        
        # Routes should be sorted by feasibility score (descending)
        scores = [route["feasibility_score"] for route in result["routes"]]
        assert scores[0] >= scores[1]  # First route should have higher or equal score


class TestConverterEdgeCases:
    """Test edge cases and error handling."""
    
    def test_recipe_with_no_operations(self):
        """Test handling of recipe with no operations field."""
        mp_recipes = [
            {
                "recipe_id": "mp-012_1",
                "target_formula": "Al2O3",
                "precursors": [{"formula": "Al(NO3)3"}],
                "temperature_celsius": 800,
                "heating_time_hours": 5
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="Al2O3",
            constraints={}
        )
        
        assert result["success"] is True
        route = result["routes"][0]
        # Should create a generic step even without operations
        assert len(route["steps"]) >= 1
    
    def test_recipe_with_no_temperature(self):
        """Test handling of recipe with no temperature data."""
        mp_recipes = [
            {
                "recipe_id": "mp-013_1",
                "target_formula": "CaO",
                "precursors": [{"formula": "CaCO3"}],
                "operations": "Heat until decomposition"
            }
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="CaO",
            constraints={}
        )
        
        assert result["success"] is True
        route = result["routes"][0]
        # Should still generate a route
        assert "temperature_range" in route
    
    def test_multiple_recipes_with_filtering(self):
        """Test conversion and filtering of multiple recipes."""
        mp_recipes = [
            {
                "recipe_id": f"mp-014_{i}",
                "target_formula": "Fe2O3",
                "precursors": [{"formula": "Fe(NO3)3"}],
                "operations": f"Heat at {temp}°C",
                "temperature_celsius": temp,
                "heating_time_hours": 5
            }
            for i, temp in enumerate([700, 900, 1100, 600], 1)
        ]
        
        result = convert_mp_recipes_to_synthesis_routes(
            mp_recipes=mp_recipes,
            target_composition="Fe2O3",
            constraints={"max_temperature": 1000}
        )
        
        assert result["success"] is True
        # Should filter out the 1100°C recipe
        assert result["n_routes"] == 3
        # All remaining routes should be under max temp
        for route in result["routes"]:
            temp_str = route["temperature_range"]
            if "°C" in temp_str:
                temp = float(temp_str.split("°C")[0])
                assert temp <= 1000
