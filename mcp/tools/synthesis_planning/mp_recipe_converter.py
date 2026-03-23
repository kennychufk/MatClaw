"""
Tool for converting Materials Project synthesis recipes to standardized route format.

This tool bridges Materials Project data with MatClaw's synthesis route format,
enabling seamless integration of literature-derived recipes into experiment planning.
"""

from typing import Dict, Any, List, Optional, Annotated
from pydantic import Field


def convert_mp_recipes_to_synthesis_routes(
    mp_recipes: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "List of synthesis recipes from Materials Project (mp_search_recipe output).\n"
                "Each recipe should contain fields like: target_formula, precursors, operations, "
                "temperature_celsius, heating_time_hours, doi, citation, etc.\n"
                "Example: Output from mp_search_recipe tool"
            )
        )
    ],
    target_composition: Annotated[
        str,
        Field(
            description=(
                "Target material composition for filtering and validation.\n"
                "Example: 'LiFePO4', 'LiCoO2'"
            )
        )
    ],
    constraints: Annotated[
        Optional[Dict[str, Any]],
        Field(
            default=None,
            description=(
                "Optional constraints for filtering recipes:\n"
                "- 'max_temperature': Maximum temperature in °C\n"
                "- 'max_time': Maximum total time in hours\n"
                "Example: {'max_temperature': 800, 'max_time': 24}"
            )
        )
    ] = None,
) -> Dict[str, Any]:
    """
    Convert Materials Project synthesis recipes to standardized synthesis route format.
    
    Takes raw recipe data from Materials Project and transforms it into the standard
    synthesis route format used throughout MatClaw. This enables literature-derived
    recipes to be used interchangeably with template-generated routes.
    
    Use this tool to:
        - Transform MP recipe data into actionable synthesis protocols
        - Filter recipes based on experimental constraints
        - Standardize heterogeneous literature data
        - Extract key synthesis parameters consistently
    
    Returns
    -------
    {
        "success": bool,
        "target_composition": str,
        "n_routes": int,
        "routes": List[Dict],  # Standardized synthesis routes
        "filtered_count": int,  # Number of recipes filtered out
        "warnings": List[str],
        "error": Optional[str]
    }
    
    Each route contains:
        - route_id: Unique identifier
        - source: "literature"
        - method: Synthesis method (solid_state, hydrothermal, etc.)
        - confidence: High confidence for literature routes (0.85-0.95)
        - feasibility_score: Overall feasibility (0-1)
        - precursors: List of starting materials
        - steps: Sequential synthesis steps with parameters
        - temperature_range: Temperature information
        - total_time_estimate: Time estimate
        - citation: Literature citation
        - doi: DOI reference
        - year: Publication year
    """
    
    # Validate inputs
    if not isinstance(mp_recipes, list):
        return {
            "success": False,
            "error": f"mp_recipes must be a list, got {type(mp_recipes).__name__}"
        }
    
    # Empty recipe list is valid - just return empty results with warning
    if not mp_recipes:
        return {
            "success": True,
            "target_composition": target_composition,
            "n_routes": 0,
            "routes": [],
            "filtered_count": 0,
            "warnings": ["No recipes provided. Try mp_search_recipe to find literature routes."]
        }
    
    # Apply constraints
    if constraints is None:
        constraints = {}
    
    max_temp = constraints.get("max_temperature", float('inf'))
    max_time = constraints.get("max_time", float('inf'))
    
    # Convert recipes
    routes = []
    filtered_count = 0
    warnings = []
    
    try:
        from pymatgen.core import Composition
        
        # Parse target composition for validation
        try:
            target_comp = Composition(target_composition)
        except Exception as e:
            warnings.append(f"Could not parse target composition: {e}")
            target_comp = None
        
    except ImportError:
        warnings.append("pymatgen not available. Composition validation disabled.")
        target_comp = None
    
    for idx, recipe in enumerate(mp_recipes, 1):
        try:
            # Extract basic information
            temperature = recipe.get("temperature_celsius")
            time_hours = recipe.get("heating_time_hours")
            
            # Apply constraint filters
            if temperature and temperature > max_temp:
                filtered_count += 1
                continue
            if time_hours and time_hours > max_time:
                filtered_count += 1
                continue
            
            # Extract precursors
            precursors_data = recipe.get("precursors", [])
            precursors = _extract_precursors(precursors_data)
            
            # Extract synthesis steps
            operations = recipe.get("operations")
            steps = _extract_steps(operations, temperature, time_hours)
            
            # Determine synthesis method
            method = _infer_synthesis_method(recipe)
            
            # Calculate scores
            confidence = 0.90  # Literature routes have high confidence
            feasibility = _calculate_feasibility_score(
                temperature or 800,
                time_hours or 12,
                max_temp,
                max_time
            )
            
            # Build standardized route
            route = {
                "route_id": idx - filtered_count,
                "source": "literature",
                "method": method,
                "confidence": confidence,
                "feasibility_score": feasibility,
                "precursors": precursors,
                "steps": steps,
                "temperature_range": f"{temperature}°C" if temperature else "See steps",
                "total_time_estimate": f"~{time_hours:.1f} hours" if time_hours else "See steps",
                "atmosphere_required": recipe.get("atmosphere") or "not specified",
                "basis": "Literature-derived from Materials Project",
                "citation": recipe.get("citation"),
                "doi": recipe.get("doi"),
                "year": recipe.get("year"),
                "recipe_id": recipe.get("recipe_id")
            }
            
            routes.append(route)
            
        except Exception as e:
            warnings.append(f"Failed to convert recipe {idx}: {str(e)}")
            continue
    
    # Generate result
    if not routes:
        return {
            "success": False,
            "target_composition": target_composition,
            "n_routes": 0,
            "routes": [],
            "filtered_count": filtered_count,
            "error": "No recipes could be converted. Check constraints or recipe format.",
            "warnings": warnings
        }
    
    return {
        "success": True,
        "target_composition": target_composition,
        "n_routes": len(routes),
        "routes": routes,
        "filtered_count": filtered_count,
        "warnings": warnings if warnings else None,
        "message": f"Successfully converted {len(routes)} recipe(s) from Materials Project"
    }


def _extract_precursors(precursors_data: Any) -> List[Dict[str, Any]]:
    """Extract and standardize precursor information."""
    precursors = []
    
    if not precursors_data:
        return precursors
    
    if isinstance(precursors_data, list):
        for prec in precursors_data:
            if isinstance(prec, dict):
                precursors.append({
                    "compound": prec.get("formula") or prec.get("name") or prec.get("material_string") or str(prec),
                    "amount": prec.get("amount"),
                    "form": prec.get("form") or "unspecified",
                    "purity": prec.get("purity")
                })
            else:
                precursors.append({
                    "compound": str(prec),
                    "amount": None,
                    "form": "unspecified"
                })
    
    return precursors


def _extract_steps(operations: Any, temperature: Optional[float], time_hours: Optional[float]) -> List[Dict[str, Any]]:
    """Extract and standardize synthesis steps from operations."""
    steps = []
    
    if operations:
        if isinstance(operations, str):
            # Parse string description into steps
            steps = _parse_operations_string(operations, temperature, time_hours)
        elif isinstance(operations, list):
            # List of operation dictionaries
            for i, op in enumerate(operations, 1):
                if isinstance(op, dict):
                    steps.append({
                        "step": i,
                        "action": op.get("type") or op.get("action") or "process",
                        "description": op.get("description") or str(op),
                        "temperature_c": op.get("temperature"),
                        "duration": op.get("duration") or op.get("time"),
                        "conditions": op.get("conditions")
                    })
                else:
                    steps.append({
                        "step": i,
                        "action": "process",
                        "description": str(op)
                    })
        else:
            # Single operation
            steps = [{
                "step": 1,
                "action": "synthesis",
                "description": str(operations)
            }]
    
    # If no detailed operations, create generic step
    if not steps:
        steps = [{
            "step": 1,
            "action": "synthesis",
            "description": f"Synthesis at {temperature}°C for {time_hours} hours" if temperature and time_hours else "Follow synthesis procedure",
            "temperature_c": temperature,
            "duration_h": time_hours
        }]
    
    return steps


def _parse_operations_string(operations: str, temperature: Optional[float], time_hours: Optional[float]) -> List[Dict[str, Any]]:
    """Parse operations text into structured steps."""
    import re
    
    steps = []
    
    # Split by common delimiters
    sentences = operations.replace(". ", ".\n").replace("; ", ";\n").split("\n")
    
    for i, sentence in enumerate(sentences, 1):
        sentence = sentence.strip()
        if not sentence:
            continue
        
        step = {
            "step": i,
            "action": "process",
            "description": sentence
        }
        
        # Try to extract temperature from text
        if "°C" in sentence or "celsius" in sentence.lower():
            temp_match = re.search(r'(\d+)\s*°?C', sentence)
            if temp_match:
                step["temperature_c"] = float(temp_match.group(1))
        
        # Try to extract time from text
        if "hour" in sentence.lower() or "hr" in sentence.lower():
            time_match = re.search(r'(\d+\.?\d*)\s*(hour|hr|h)', sentence.lower())
            if time_match:
                step["duration_h"] = float(time_match.group(1))
        
        steps.append(step)
    
    # If no steps extracted, create one
    if not steps:
        steps = [{
            "step": 1,
            "action": "synthesis",
            "description": operations,
            "temperature_c": temperature,
            "duration_h": time_hours
        }]
    
    return steps


def _infer_synthesis_method(recipe: Dict[str, Any]) -> str:
    """Infer synthesis method from recipe metadata."""
    
    # Check atmosphere and conditions
    atmosphere = str(recipe.get("atmosphere", "")).lower()
    conditions = str(recipe.get("conditions", "")).lower()
    operations = str(recipe.get("operations", "")).lower()
    
    # Look for method indicators
    all_text = f"{atmosphere} {conditions} {operations}"
    
    if "hydrothermal" in all_text or "autoclave" in all_text:
        return "hydrothermal"
    elif "solution" in all_text or "precipitation" in all_text:
        return "solution"
    elif "sol-gel" in all_text or "sol_gel" in all_text:
        return "sol_gel"
    elif "combustion" in all_text:
        return "combustion"
    elif "melt" in all_text:
        return "melting"
    else:
        return "solid_state"  # Default


def _calculate_feasibility_score(
    temperature: float,
    time_hours: float,
    max_temp: float,
    max_time: float
) -> float:
    """Calculate feasibility score for a literature route."""
    
    score = 1.0
    
    # Penalize if approaching limits
    if temperature > max_temp * 0.9:
        score -= 0.15
    if time_hours > max_time * 0.9:
        score -= 0.15
    
    # Literature routes get bonus for being proven
    score += 0.10  # Proven in literature
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))
