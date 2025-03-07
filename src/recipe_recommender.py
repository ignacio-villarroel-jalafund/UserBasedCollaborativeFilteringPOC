import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class RecipeRecommender:
    
    def __init__(self, data_processor):
        """
        Inicializa el recomendador de recetas.
        
        Args:
            data_processor (DataProcessor): Instancia de DataProcessor con los datos procesados.
        """
        self.data_processor = data_processor
        self.user_item_matrix = None
        self.user_similarity = None
    
    def load_user_preferences(self, user_likes_data):
        """
        Carga las preferencias de los usuarios y construye la matriz usuario-item.
        
        Args:
            user_likes_data (dict o DataFrame): Datos de preferencias de los usuarios.
        """
        self.user_likes = pd.DataFrame(user_likes_data)
        self._build_user_item_matrix()
        self._compute_user_similarity()
    
    def _build_user_item_matrix(self):
        """Construye la matriz usuario-item para el filtrado colaborativo."""
        recipe_ids = self.data_processor.recipes['Id'].unique()
        self.user_item_matrix = self.user_likes.pivot_table(
            index='user_id', 
            columns='recipe_id', 
            values='like', 
            fill_value=0
        )
        self.user_item_matrix = self.user_item_matrix.reindex(columns=recipe_ids, fill_value=0)
    
    def _compute_user_similarity(self):
        """Calcula la similitud entre usuarios usando similitud del coseno."""
        similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            similarity, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
    
    def recommend_recipes(self, target_user, user_allergies, top_n=5):
        """
        Genera recomendaciones de recetas para un usuario objetivo.
        
        Args:
            target_user (int): ID del usuario para el que se generan recomendaciones.
            user_allergies (list): Lista de alergias del usuario.
            top_n (int): Número de recomendaciones a generar.
            
        Returns:
            DataFrame: Recetas recomendadas.
        """
        similar_users = self.user_similarity.loc[target_user].drop(target_user)
        similar_users = similar_users[similar_users > 0]
        
        if similar_users.empty:
            print("No se encontraron usuarios similares para generar recomendaciones.")
            return pd.DataFrame()
        
        """ 
        Calcular puntuaciones

        """
        weighted_likes = self.user_item_matrix.loc[similar_users.index].multiply(similar_users, axis=0).sum()
        already_liked = self.user_item_matrix.loc[target_user]
        candidate_scores = weighted_likes[already_liked == 0]
        
        """"
        Filtrar recetas con posibles alergias
        """
        safe_recipe_ids = []
        for recipe_id in candidate_scores.index:
            recipe = self.data_processor.get_recipe_by_id(recipe_id)
            if not recipe.empty:
                recipe_ingredients = recipe['Cleaned_Ingredients'].values[0]
                if self.data_processor.is_recipe_safe(recipe_ingredients, user_allergies):
                    safe_recipe_ids.append(recipe_id)
        
        if not safe_recipe_ids:
            return pd.DataFrame()
        
        """ 
        Ordenar y seleccionar las mejores recomendaciones
        """
        safe_scores = candidate_scores.loc[safe_recipe_ids]
        recommended_ids = safe_scores.sort_values(ascending=False).head(top_n).index.tolist()
        
        """ 
        Obtener detalles de las recetas recomendadas"
        """
        recommendations = self.data_processor.recipes[
            self.data_processor.recipes['Id'].isin(recommended_ids)
        ]
        return recommendations
