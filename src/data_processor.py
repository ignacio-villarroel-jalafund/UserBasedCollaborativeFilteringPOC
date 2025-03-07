import pandas as pd
import ast

class DataProcessor:
    """Clase para cargar y procesar los datos de recetas y alergias."""
    
    def __init__(self, food_data_path, recipes_path):
        """
        Inicializa el procesador de datos.
        
        Args:
            food_data_path (str): Ruta al archivo CSV con datos de alimentos y alergias.
            recipes_path (str): Ruta al archivo CSV con datos de recetas.
        """
        self.food_data = pd.read_csv(food_data_path)
        self.recipes = pd.read_csv(recipes_path)
        self.allergy_mapping = self._create_allergy_mapping()
        self._preprocess_recipes()
    
    def _create_allergy_mapping(self):
        """Crea un diccionario que mapea alergias a alimentos."""
        return self.food_data.groupby('Allergy')['Food'].apply(list).to_dict()
    
    def _preprocess_recipes(self):
        """Preprocesa los ingredientes de las recetas."""
        self.recipes['Cleaned_Ingredients'] = self.recipes['Cleaned_Ingredients'].apply(self._process_ingredients)
    
    def _process_ingredients(self, ingredients_str):
        """
        Convierte la cadena de ingredientes en un formato consistente.
        
        Args:
            ingredients_str (str): Cadena de texto con los ingredientes.
            
        Returns:
            str: Cadena de texto con los ingredientes procesados.
        """
        try:
            ingredients_list = ast.literal_eval(ingredients_str)
            return " ".join(ingredients_list)
        except Exception:
            return ingredients_str
    
    def is_recipe_safe(self, recipe_ingredients, user_allergies):
        """
        Verifica si una receta es segura para un usuario con alergias.
        
        Args:
            recipe_ingredients (str): Ingredientes de la receta.
            user_allergies (list): Lista de alergias del usuario.
            
        Returns:
            bool: True si la receta es segura, False si contiene alérgenos.
        """
        text = recipe_ingredients.lower()
        for allergy in user_allergies:
            if allergy in self.allergy_mapping:
                for food in self.allergy_mapping[allergy]:
                    if food.lower() in text:
                        return False
        return True
    
    def get_recipe_by_id(self, recipe_id):
        """Obtiene una receta por su ID."""
        return self.recipes[self.recipes['Id'] == recipe_id]
