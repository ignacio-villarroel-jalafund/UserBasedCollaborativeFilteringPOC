from data_processor import DataProcessor
from recipe_recommender import RecipeRecommender

def main():
    """Función principal que ejecuta el sistema de recomendación."""
    FOOD_DATA_PATH = './datasets/FoodData.csv'
    RECIPES_PATH = './datasets/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
    
    user_likes_data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        'recipe_id': [0, 1, 2, 1, 3, 5, 0, 3, 6, 2, 3, 8, 9],
        'like': [1] * 13
    }
    
    """ 
    Citrus Allergy
    """
    TARGET_USER = 1
    USER_ALLERGIES = ['Nut Allergy']
    TOP_N_RECOMMENDATIONS = 5
    
    data_processor = DataProcessor(FOOD_DATA_PATH, RECIPES_PATH)
    recommender = RecipeRecommender(data_processor)
    recommender.load_user_preferences(user_likes_data)
    
    recommendations = recommender.recommend_recipes(
        TARGET_USER, 
        USER_ALLERGIES, 
        TOP_N_RECOMMENDATIONS
    )
    
    if recommendations.empty:
        print("No se encontraron recetas seguras para recomendar al usuario.")
    else:
        print(f"Recetas recomendadas para el usuario (ID {TARGET_USER}):")
        print(recommendations[['Id', 'Title', 'Cleaned_Ingredients']])
    
if __name__ == '__main__':
    main()
