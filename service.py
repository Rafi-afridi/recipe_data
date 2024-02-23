# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import torch
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load tokenizer from file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer_lstm = pickle.load(handle)

# Load model from file
model_lstm = load_model('model.h5')

max_sequence_len = 37

# Define a function for generating responses
def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Explicitly set attention_mask and pad_token_id
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id  # or any other suitable token
    
    output = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7, attention_mask=attention_mask, pad_token_id=pad_token_id)
    
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_response

all_ingredients = pd.read_csv("ingredients.csv")
recipe_data = pd.read_csv("random_recipes.csv")
recipe_data['directions'] = recipe_data['directions'].apply(eval)  
recipe_data['ingredients'] = recipe_data['ingredients'].apply(eval)  
df_small = pd.read_csv("recom_system.csv")
titles = df_small['title']
indices = pd.Series(df_small.index, index=df_small['title'])
cosine_sim = np.load("cosine_sim.npy")

def show_most_related_ingredients(ingredient):
    id_list = all_ingredients[all_ingredients.ingredient == ingredient]["title"].unique()
    # Find all rows for the above id's and do value_counts on those rows
    return all_ingredients[all_ingredients.title.isin(id_list)].ingredient.value_counts().head(10).index.tolist()[1:]

def find_best_recipe(ingredients):
    best_recipe = None
    best_match_count = 0

    for _, recipe in recipe_data.iterrows():
        recipe_ingredients = recipe['ingredients']
        match_count = sum(1 for ingredient in ingredients if ingredient in recipe_ingredients)

        if match_count > best_match_count:
            best_recipe = recipe
            best_match_count = match_count

    title = best_recipe['title']
    directions = best_recipe['directions']
    ingredients_required = set(best_recipe['ingredients']).difference(set(ingredients))
    return title, "\n".join(directions), "\n".join(ingredients_required)
 
def get_recommendations(title, no_of_recipes=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    recipe_indices = [i[0] for i in sim_scores]
    return titles.iloc[recipe_indices].head(no_of_recipes).values.tolist()

def generate_text(prompt):
    # Generate new recipe directions
    next_words = 3
    
    for _ in range(next_words):
        token_list = tokenizer_lstm.texts_to_sequences([prompt])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')  # Modify maxlen argument
        predicted = model_lstm.predict(token_list, verbose=0)  # Replace predict_classes with predict
        
        predicted_word_index = np.argmax(predicted)
        output_word = ""
        for word, index in tokenizer_lstm.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        prompt += " " + output_word
    
    return "Generated Directions:"+prompt
    
# Define Streamlit App
def main():
    st.sidebar.title('Use Cases')
    project_option = st.sidebar.radio('', ['GPT-2 Model for Recipe QA', 'Ingredients Semantic Network', 'Recipe Recommendation (Ingredient Based)', 'Recipe Recommendation (Cosine Similarity)', 'Recipe Assistant (Text Generation)'])

    if project_option == 'GPT-2 Model for Recipe QA':
        project_1()
    elif project_option == 'Ingredients Semantic Network':
        project_2()
    elif project_option == 'Recipe Recommendation (Ingredient Based)':
        project_3()
    elif project_option == 'Recipe Recommendation (Cosine Similarity)':
        project_4()
    elif project_option == 'Recipe Assistant (Text Generation)':
        project_5()
        
# Project 1 Page
def project_1():
    st.title('GPT-2 Model for Recipe QA')
    st.write("GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data.")

    prompt = st.text_area("Ask me anything about Recipes:")
    if st.button("Submit"):
        if prompt:
            response = generate_response(prompt)
            st.markdown(f'<p style="color:darkgreen">{response}</p>', unsafe_allow_html=True)

# Project 2 Page
def project_2():
    st.title('Ingredients Semantic Network')
    st.write("Here you can see the most used ingredients together")
    items = all_ingredients.ingredient.values.tolist()
    selected_value = st.selectbox("Select a value", items)
    if st.button("Submit"):
        if selected_value:
            response_list = show_most_related_ingredients(selected_value)
            st.write(f"Following are Most Frequent Ingredients used with {selected_value}:")
            st.markdown("<ul>" + "".join([f"<li>{item}</li>" for item in response_list]) + "</ul>", unsafe_allow_html=True)

# Project 3 Page (To be implemented)
def project_3():
    st.title('What Should I Cook?')
    st.write("This is Similarity Based Recommendation System, where system will recommend recipe based on what ingredients you have and whats missing.")
    
    selected_ingredients = st.multiselect("Select Ingredients", list(recipe_data['ingredients'].explode().unique()))
    if st.button("Submit"):
        if selected_ingredients:
            title, directions, ingredients_required = find_best_recipe(selected_ingredients)
            st.write("Best Recipe Found:")
            st.write(f"Title: {title}")
            st.write("Directions:")
            st.write(directions)
            st.write("Additional Ingredients Required:")
            st.write(ingredients_required)

# Project 4 Page
def project_4():
    st.title('Recommendation System (Cosine Similarity)')
    st.write("")
    selected_value = st.selectbox("Select a Recipe", titles.values.tolist())
    if st.button("Submit"):
        if selected_value:
            response_list = get_recommendations(selected_value)
            st.write(f"You might like following recipes, recommended against {selected_value}:")
            st.markdown("<ul>" + "".join([f"<li>{item}</li>" for item in response_list]) + "</ul>", unsafe_allow_html=True)

# Project 4 Page
def project_5():
    st.title('Recipe Assistant')
    st.write("Deep Learning/ LSTM/ RNN Based Text Generation")
    st.write("Enter text such as  {In a large bowl, mix}")

    prompt = st.text_area("Enter initial steps to make recipe:")
    if st.button("Submit"):
        if prompt:
            response = generate_text(prompt)
            st.markdown(f'<p style="color:darkgreen">{response}</p>', unsafe_allow_html=True) 
            
if __name__ == "__main__":
    main()
