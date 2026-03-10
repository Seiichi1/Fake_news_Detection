import pandas as pd
import random

def generate_synthetic_data(num_samples=500):
    subjects = [
        "Aliens", "Bigfoot", "The Loch Ness Monster", "A time traveler", "The Illuminati",
        "NASA", "The Government", "Scientists", "Elon Musk", "A secret society",
        "Zombies", "Vampires", "Ghosts", "The President", "A Florida Man"
    ]
    
    verbs = [
        "discovered", "revealed", "admitted to", "captured", "ate",
        "destroyed", "invented", "banned", "replaced", "confirmed",
        "found", "hiding", "selling", "communicating with", "worshipping"
    ]
    
    objects = [
        "a city on Mars", "the cure for death", "pizza", "all birds with drones", "the moon",
        "a portal to hell", "mind control devices", "infinite energy", "dinosaurs", "Atlantis",
        "human clones", "5G towers", "the internet", "gravity", "water"
    ]
    
    locations = [
        "in New York City", "at Area 51", "under the ocean", "on the dark side of the moon",
        "in a secret bunker", "at the White House", "in Antarctica", "inside a volcano",
        "in a Walmart", "on live TV"
    ]
    
    data = []
    
    print(f"Generating {num_samples} synthetic fake news samples...")
    
    for _ in range(num_samples):
        # Template 1: Subject Verb Object Location
        if random.random() < 0.5:
            text = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} {random.choice(locations)}."
        # Template 2: BREAKING: Subject Verb Object
        elif random.random() < 0.5:
            text = f"BREAKING: {random.choice(subjects)} has {random.choice(verbs)} {random.choice(objects)}!"
        # Template 3: Subject confirms Object is Verb
        else:
            text = f"{random.choice(subjects)} confirms that {random.choice(objects)} is actually {random.choice(verbs)} by {random.choice(subjects)}."
            
        # Add some "clickbait" style
        if random.random() < 0.3:
            text += " You won't believe what happens next!"
        if random.random() < 0.3:
            text = text.upper()
            
        data.append({'text': text, 'label': 1}) # Label 1 = Fake (CORRECTED)
        
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = 'data/synthetic_fake_news.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Preview
    print("\n--- Preview ---")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_data()
