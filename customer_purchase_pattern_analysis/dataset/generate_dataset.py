"""Generates dataset/supermarket_transactions.csv (seeded -> reproducible).
Run:  python dataset/generate_dataset.py
The generated CSV is already included in the project; this script is only
provided so you can explain / reproduce how the data was produced."""
import os, numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supermarket_transactions.csv")
N_TX, N_CUST = 35000, 3000
rng = np.random.default_rng(2024)

CATALOG = """
Fresh Fruits|Bananas 1kg:1.2;Apples Red 1kg:2.8;Oranges 1kg:2.5;Grapes Green 500g:3.4;Mangoes 1kg:3.9;Pears 1kg:2.9;Pineapple Whole:2.6;Watermelon Whole:4.5;Kiwi 4pk:2.4
Citrus Fruits|Lemons 500g:1.8;Limes 250g:1.5;Grapefruit 1kg:2.7;Mandarins 1kg:3.1
Tropical Fruits|Papaya Whole:2.9;Pomegranate 2pk:3.3;Fresh Coconut:1.9;Dragon Fruit 2pk:4.2
Berries|Strawberries 400g:3.6;Blueberries 250g:3.9;Raspberries 170g:3.8;Blackberries 170g:3.7
Dried Fruits|Raisins 500g:3.2;Dates 500g:4.1;Dried Apricots 250g:3.5;Prunes 400g:3.4
Leafy Vegetables|Spinach 250g:1.6;Lettuce Iceberg:1.4;Kale Bunch:1.9;Cabbage Green:1.3
Root Vegetables|Potatoes 2kg:3.2;Carrots 1kg:1.5;Sweet Potatoes 1kg:2.6;Beetroot 500g:1.7
Fresh Vegetables|Tomatoes 1kg:2.4;Cucumbers 3pk:1.8;Bell Peppers 3pk:3.0;Broccoli Head:1.9;Zucchini 500g:2.0;Cauliflower Head:2.2
Herbs & Aromatics|Garlic Bulbs 3pk:1.4;Ginger Root 250g:1.6;Onions Yellow 1kg:1.7;Fresh Coriander Bunch:0.9;Fresh Basil Pot:2.1;Fresh Mint Bunch:1.0
Mushrooms|Button Mushrooms 250g:2.3;Portobello Mushrooms 200g:3.0;Shiitake Mushrooms 150g:3.8
Salad Kits|Caesar Salad Kit:3.9;Garden Salad Mix 300g:2.9;Coleslaw Mix 400g:2.4
Milk|Whole Milk 1L:1.3;Skim Milk 1L:1.3;Semi-Skimmed Milk 2L:2.2;Lactose-Free Milk 1L:1.9
Cheese|Cheddar Cheese 200g:3.1;Mozzarella Cheese 250g:2.9;Parmesan Cheese 100g:3.8;Cream Cheese 200g:2.4;Sliced Swiss Cheese 150g:3.3
Yogurt|Greek Yogurt 500g:2.9;Fruit Yogurt 4pk:2.7;Plain Yogurt 1kg:2.5;Drinking Yogurt 6pk:3.6
Butter & Margarine|Salted Butter 250g:2.9;Unsalted Butter 250g:3.0;Margarine Tub 500g:2.1
Cream|Whipping Cream 250ml:2.2;Sour Cream 200ml:1.7;Cooking Cream 250ml:1.9
Eggs|Eggs Large 12pk:3.1;Eggs Free Range 6pk:2.6;Eggs Organic 12pk:4.4
Plant-Based Milk|Almond Milk 1L:2.4;Soy Milk 1L:2.0;Oat Milk 1L:2.3;Coconut Milk Drink 1L:2.6
Bread|White Bread Loaf:1.8;Whole Wheat Bread Loaf:2.2;Multigrain Bread Loaf:2.6;Sourdough Loaf:3.5;Garlic Bread:2.3
Buns & Rolls|Burger Buns 6pk:2.1;Hot Dog Rolls 6pk:2.0;Dinner Rolls 8pk:2.4;Croissants 4pk:3.2
Cakes & Pastries|Chocolate Cake Slice:2.8;Blueberry Muffins 4pk:3.4;Donuts 6pk:3.9;Apple Pie:4.6
Cookies & Biscuits|Tea Biscuits 300g:1.9;Chocolate Chip Cookies 250g:2.6;Digestive Biscuits 400g:2.2;Cream Sandwich Biscuits 300g:2.0
Tortillas & Wraps|Flour Tortillas 8pk:2.5;Corn Tortillas 12pk:2.4;Pita Bread 6pk:1.9
Bakery Mixes|Cake Mix Vanilla:2.3;Brownie Mix:2.5;Bread Flour Mix:2.9
Chicken|Chicken Breast 1kg:7.2;Chicken Thighs 1kg:5.4;Whole Chicken 1.5kg:6.9;Chicken Wings 1kg:5.9;Chicken Drumsticks 1kg:4.8
Beef|Ground Beef 500g:5.6;Beef Steak Sirloin 300g:8.9;Beef Stew Meat 500g:6.8;Beef Burger Patties 4pk:4.9
Pork|Pork Chops 500g:5.1;Pork Mince 500g:4.4;Pork Ribs 1kg:8.2
Lamb & Mutton|Lamb Chops 500g:9.5;Lamb Mince 500g:7.2;Mutton Curry Cut 1kg:11.0
Deli Ham & Cold Cuts|Sliced Ham 200g:3.2;Turkey Slices 150g:3.0;Salami 100g:2.9;Roast Beef Slices 150g:3.8
Sausages & Bacon|Pork Sausages 400g:3.7;Streaky Bacon 250g:3.6;Chicken Sausages 400g:3.9;Frankfurters 8pk:2.8
Fish|Salmon Fillet 300g:7.5;Tilapia Fillets 400g:5.2;Cod Fillet 300g:6.4;Fresh Tuna Steak 250g:7.9;Mackerel 2pk:4.5
Shrimp & Prawns|Raw Shrimp 400g:8.3;Cooked Prawns 250g:6.9;Tiger Prawns 500g:10.5
Shellfish|Mussels 1kg:5.5;Squid Rings 300g:5.0;Crab Meat 170g:6.2
Canned Fish|Canned Tuna 3pk:3.9;Canned Sardines 2pk:2.8;Canned Salmon 213g:3.4
Rice|Basmati Rice 1kg:3.2;Jasmine Rice 1kg:2.9;Brown Rice 1kg:3.0;Long Grain Rice 2kg:3.8;Arborio Rice 500g:2.7
Pasta|Spaghetti 500g:1.6;Penne Pasta 500g:1.6;Fusilli Pasta 500g:1.7;Lasagna Sheets 250g:2.1;Macaroni 500g:1.5
Pasta Sauce|Tomato Pasta Sauce 400g:2.2;Bolognese Sauce 400g:2.6;Pesto Sauce 190g:2.9;Alfredo Sauce 400g:2.8
Noodles|Instant Noodles 5pk:2.6;Egg Noodles 400g:2.1;Rice Noodles 400g:2.3;Udon Noodles 2pk:2.9
Flour|All Purpose Flour 1kg:1.6;Whole Wheat Flour 2kg:2.4;Cornflour 500g:1.5
Lentils & Pulses|Red Lentils 1kg:2.8;Chickpeas Dried 1kg:2.7;Black Beans Dried 500g:2.1;Green Peas Dried 500g:1.9
Breakfast Cereal|Corn Flakes 500g:3.2;Chocolate Cereal 375g:3.9;Muesli 750g:4.5;Rolled Oats 1kg:2.9
Breakfast Spreads|Strawberry Jam 340g:2.6;Peanut Butter 340g:3.1;Honey 350g:4.2;Chocolate Hazelnut Spread 350g:3.9;Marmalade 340g:2.4
Pancake & Waffle|Pancake Mix 500g:2.5;Maple Syrup 250ml:4.8;Waffle Mix 500g:2.7
Coffee|Instant Coffee 200g:6.9;Ground Coffee 250g:5.8;Coffee Beans 500g:8.5;Coffee Pods 16pk:5.9
Tea|Black Tea Bags 100pk:3.2;Green Tea Bags 25pk:2.4;Herbal Tea 20pk:2.7;Chai Tea 50pk:3.5
Sugar & Sweeteners|White Sugar 1kg:1.7;Brown Sugar 500g:1.9;Sweetener Tablets 300:2.2;Icing Sugar 500g:1.8
Salt & Pepper|Table Salt 1kg:0.8;Sea Salt Grinder:2.4;Black Pepper Grinder:2.6
Spices|Turmeric Powder 100g:1.9;Cumin Seeds 100g:2.0;Paprika 80g:1.8;Cinnamon Sticks 40g:2.2;Chili Powder 100g:1.9;Garam Masala 100g:2.4
Cooking Oil|Sunflower Oil 1L:3.2;Olive Oil Extra Virgin 500ml:6.9;Vegetable Oil 2L:4.5;Coconut Oil 500ml:5.6
Vinegar & Dressings|White Vinegar 500ml:1.3;Balsamic Vinegar 250ml:3.4;Ranch Dressing 350ml:2.9;Italian Dressing 350ml:2.8
Ketchup & Sauces|Tomato Ketchup 500g:2.3;BBQ Sauce 350g:2.7;Hot Sauce 150ml:2.4;Soy Sauce 250ml:2.3
Condiments|Mayonnaise 400g:2.8;Yellow Mustard 250g:1.8;Relish 300g:2.1
Canned Vegetables|Canned Tomatoes 400g:1.2;Sweet Corn Can 340g:1.4;Canned Mushrooms 400g:1.7;Canned Green Peas 400g:1.3
Canned Beans & Soup|Baked Beans 415g:1.4;Kidney Beans Can 400g:1.3;Chicken Noodle Soup Can:1.9;Tomato Soup Can:1.8
Instant Meals & Soups|Instant Soup Sachets 4pk:2.1;Mac and Cheese Box:2.0;Instant Mashed Potato:2.2
Pickles & Olives|Green Olives 340g:2.9;Pickled Cucumbers 500g:2.4;Capers 100g:2.5
Baking Essentials|Baking Powder 100g:1.6;Baking Soda 200g:1.3;Vanilla Extract 50ml:3.9;Yeast Sachets 3pk:1.5;Cocoa Powder 250g:3.8
Chocolate|Milk Chocolate Bar:1.8;Dark Chocolate Bar:2.2;Chocolate Box Assorted:5.5;White Chocolate Bar:1.9
Candy & Gum|Gummy Bears 200g:1.9;Mint Chewing Gum:1.2;Hard Candy Bag:1.8;Lollipops 10pk:2.1
Chips & Crisps|Potato Chips Salted:2.1;Tortilla Chips 300g:2.6;Cheese Puffs 150g:1.9;BBQ Potato Chips:2.1
Popcorn & Puffed|Microwave Popcorn 3pk:2.4;Popcorn Kernels 500g:2.0;Rice Cakes 150g:1.9
Nuts & Seeds|Almonds Roasted 200g:4.5;Cashew Nuts 200g:5.2;Peanuts Salted 400g:2.8;Pistachios 150g:5.6;Sunflower Seeds 250g:2.4
Crackers|Cream Crackers 300g:1.8;Cheese Crackers 200g:2.3;Water Crackers 125g:2.0
Granola Bars|Granola Bars 6pk:3.1;Protein Bars 4pk:5.2;Fruit Bars 6pk:3.0
Soft Drinks|Cola 2L:2.1;Lemon Lime Soda 2L:2.0;Orange Soda 2L:2.0;Ginger Ale 1.25L:1.9;Cola Cans 12pk:6.2
Bottled Water|Mineral Water 6x1.5L:3.9;Spring Water 24x500ml:5.5;Alkaline Water 1L:1.6
Sparkling Water|Sparkling Water 6pk:3.4;Flavored Sparkling Lemon 6pk:3.9;Tonic Water 4pk:3.2
Energy Drinks|Energy Drink 250ml:1.9;Energy Drink 4pk:6.9;Sugar-Free Energy Drink:2.0
Sports Drinks|Electrolyte Drink 500ml:1.8;Isotonic Drink 6pk:7.2;Protein Shake 330ml:2.9
Iced Tea & Coffee RTD|Iced Tea Peach 1.5L:2.1;Cold Brew Coffee 250ml:2.6;Iced Latte Bottle:2.4
Fruit Juice|Orange Juice 1L:2.9;Apple Juice 1L:2.6;Mango Juice 1L:2.8;Cranberry Juice 1L:3.2
Beer|Lager Beer 6pk:8.9;Craft IPA 4pk:9.5;Light Beer 12pk:13.5
Wine|Red Wine 750ml:9.9;White Wine 750ml:9.5;Sparkling Wine 750ml:12.9
Spirits & Mixers|Vodka 750ml:18.0;Whisky 700ml:24.0;Rum 700ml:17.5
Frozen Vegetables|Frozen Peas 900g:2.4;Frozen Mixed Vegetables 1kg:2.8;Frozen Spinach 500g:2.2;Frozen Sweet Corn 900g:2.3
Frozen Meals|Frozen Lasagna 400g:3.9;Frozen Chicken Curry 350g:3.6;Frozen Veg Biryani 350g:3.3
Frozen Pizza|Frozen Pepperoni Pizza:4.5;Frozen Margherita Pizza:3.9;Frozen Veggie Pizza:4.1
Ice Cream|Vanilla Ice Cream 1L:3.8;Chocolate Ice Cream 1L:3.9;Ice Cream Cones 6pk:3.2;Ice Cream Bars 8pk:4.9
Frozen Meat & Nuggets|Chicken Nuggets 500g:4.2;Frozen Fish Fingers 400g:3.4;Frozen Meatballs 500g:4.8
Frozen Desserts|Frozen Cheesecake:5.2;Frozen Fruit Mix 500g:3.6;Frozen Waffles 8pk:2.9
Frozen Seafood|Frozen Shrimp 500g:7.2;Frozen Fish Fillets 800g:7.9;Frozen Calamari 400g:5.6
Frozen Potato Products|French Fries 1kg:2.5;Hash Browns 8pk:2.6;Potato Wedges 750g:2.7
Dish Care|Dishwashing Liquid 750ml:2.3;Dishwasher Tablets 30pk:7.2;Dish Sponges 6pk:1.9
Laundry|Laundry Detergent 2L:8.9;Fabric Softener 1.5L:4.5;Stain Remover 500ml:3.9
Cleaning Supplies|All Purpose Cleaner 750ml:2.6;Glass Cleaner 500ml:2.4;Bleach 1L:1.9;Floor Cleaner 1L:3.2
Paper Products|Toilet Paper 12pk:6.9;Paper Towels 4pk:5.4;Facial Tissues 3pk:3.1;Paper Napkins 100pk:1.8
Trash Bags & Foil|Trash Bags 30pk:3.5;Aluminium Foil 30m:3.2;Cling Film 50m:2.8;Zip Lock Bags 40pk:3.3
Air Fresheners|Air Freshener Spray:2.9;Scented Candle:4.5;Toilet Freshener 2pk:2.8
Home Essentials|LED Light Bulbs 4pk:7.5;AA Batteries 8pk:6.5;Matches 10pk:1.0
Shampoo & Conditioner|Shampoo 400ml:4.5;Conditioner 400ml:4.5;Anti-Dandruff Shampoo 400ml:5.2
Body Wash & Soap|Body Wash 500ml:3.9;Bar Soap 4pk:3.2;Hand Wash 250ml:2.4;Hand Sanitizer 250ml:2.9
Oral Care|Toothpaste 100ml:2.6;Toothbrush 2pk:2.9;Mouthwash 500ml:3.9;Dental Floss:2.1
Deodorant|Deodorant Roll-On:3.1;Deodorant Spray:3.4;Antiperspirant Stick:3.6
Skin Care|Face Moisturizer 50ml:6.5;Face Wash 150ml:4.9;Body Lotion 400ml:4.5;Lip Balm:1.9
Hair Care Styling|Hair Gel 250ml:3.8;Hair Spray 300ml:3.9;Hair Oil 200ml:4.4
Shaving|Razor Blades 4pk:7.5;Shaving Foam 200ml:3.4;After Shave Lotion 100ml:4.9
Feminine Care|Sanitary Pads 14pk:3.4;Tampons 20pk:3.6;Panty Liners 30pk:2.4
Cosmetics|Mascara:6.9;Lipstick:7.5;Nail Polish:3.9;Makeup Remover Wipes:3.2
Sun Care|Sunscreen SPF50 200ml:8.9;After Sun Lotion 200ml:6.4;Insect Repellent:4.9
Baby Food|Baby Cereal 250g:3.6;Fruit Puree Pouches 4pk:3.9;Baby Veg Jar 4pk:3.4
Diapers & Wipes|Baby Diapers Size 3:12.9;Baby Diapers Size 4:13.5;Baby Wipes 3pk:4.5
Baby Formula|Infant Formula 800g:19.0;Follow-On Milk 800g:17.0;Toddler Milk 800g:15.5
Baby Bath & Care|Baby Shampoo 300ml:3.9;Baby Lotion 300ml:4.2;Diaper Rash Cream:5.0
Dog Food|Dry Dog Food 5kg:16.5;Wet Dog Food 6pk:7.9;Puppy Food 3kg:12.5
Cat Food|Dry Cat Food 2kg:9.9;Wet Cat Food 12pk:8.5;Kitten Food 1.5kg:8.0
Pet Treats|Dog Biscuits 500g:3.9;Dental Chews 7pk:4.5;Cat Treats 60g:2.0
Pet Litter & Care|Cat Litter 10L:8.5;Pet Shampoo 250ml:5.2;Poop Bags 60pk:3.5
Vitamins & Supplements|Multivitamin 60 tabs:9.5;Vitamin C 30 tabs:6.4;Omega-3 Capsules 60:11.0;Protein Powder 1kg:24.0
First Aid|Adhesive Bandages 40pk:3.5;Antiseptic Cream 30g:4.1;Cotton Wool 100g:2.0
Cold & Flu|Cough Syrup 150ml:5.5;Throat Lozenges 24pk:3.4;Nasal Spray:4.8
Pain Relief|Paracetamol 16 tabs:2.5;Ibuprofen 16 tabs:3.6;Heat Patch 4pk:4.9
Hummus & Dips|Hummus 200g:2.5;Guacamole 150g:2.9;Tzatziki 200g:2.4;Salsa Dip 300g:2.6
Tofu & Meat Alternatives|Firm Tofu 400g:2.5;Plant Burger Patties 2pk:4.5;Tempeh 200g:3.2
Gluten-Free|Gluten-Free Bread:3.9;Gluten-Free Pasta 500g:2.9;Gluten-Free Flour 1kg:3.5
Organic Staples|Organic Quinoa 500g:4.9;Organic Chia Seeds 250g:4.5;Organic Brown Lentils 500g:3.1
Paneer & Ghee|Paneer 200g:2.9;Ghee 500ml:6.9;Papad 200g:1.8
Asian Sauces|Sweet Chili Sauce 300ml:2.4;Teriyaki Sauce 250ml:2.8;Oyster Sauce 255ml:2.7;Curry Paste 200g:2.9
Mexican Foods|Taco Shells 12pk:2.6;Taco Seasoning:1.2;Refried Beans Can:1.9;Jalapeno Slices Jar:2.1
Gourmet Specialty|Truffle Oil 100ml:9.9;Sun-Dried Tomatoes 200g:3.9;Artichoke Hearts Jar:3.5
Chilled Ready Meals|Chilled Lasagne 400g:4.5;Chilled Chicken Tikka Masala:4.9;Sushi Box:6.5
Fresh Juice & Smoothies|Fresh Orange Juice 1L:3.9;Green Smoothie 330ml:3.5;Berry Smoothie 330ml:3.5
Syrups & Toppings|Chocolate Syrup 350g:2.6;Caramel Sauce 300g:2.9;Sprinkles 80g:1.9
Canned Fruit|Canned Peaches 410g:1.9;Canned Pineapple Slices 425g:1.8;Fruit Cocktail 410g:1.9
Party Supplies|Disposable Cups 50pk:3.0;Paper Plates 50pk:3.4;Balloons 20pk:2.5;Birthday Candles 24pk:1.9
Seasonal Items|Festive Gift Box:14.0;Gift Wrap Roll:3.9;Seasonal Candy Box:6.5
Kitchen Tools|Kitchen Knife:6.5;Plastic Containers 3pk:5.5;Wooden Spoons 3pk:3.9
Ready-to-Cook Mixes|Biryani Masala Kit:2.4;Pasta Meal Kit:3.4;Fajita Kit:4.0
Cooking Stock|Chicken Stock Cubes 12pk:1.8;Vegetable Stock Cubes 12pk:1.8;Beef Stock Pot 4pk:2.7
Smoked & Cured|Smoked Salmon 100g:4.9;Pastrami 100g:3.6;Chorizo 150g:3.5
Dairy Desserts|Chocolate Mousse 4pk:2.9;Rice Pudding 4pk:2.6;Custard Pots 4pk:2.4
"""
# (anchor, companion, probability) -> realistic but NOT deterministic associations
AFF = [
("Spaghetti 500g","Tomato Pasta Sauce 400g",.55),("Penne Pasta 500g","Tomato Pasta Sauce 400g",.50),("Fusilli Pasta 500g","Pesto Sauce 190g",.30),
("Spaghetti 500g","Bolognese Sauce 400g",.22),("Spaghetti 500g","Parmesan Cheese 100g",.22),("Lasagna Sheets 250g","Bolognese Sauce 400g",.45),("Lasagna Sheets 250g","Mozzarella Cheese 250g",.35),
("Ground Coffee 250g","White Sugar 1kg",.45),("Instant Coffee 200g","White Sugar 1kg",.42),("Instant Coffee 200g","Whole Milk 1L",.30),("Ground Coffee 250g","Semi-Skimmed Milk 2L",.30),
("Corn Flakes 500g","Whole Milk 1L",.60),("Chocolate Cereal 375g","Whole Milk 1L",.55),("Muesli 750g","Greek Yogurt 500g",.32),("Rolled Oats 1kg","Honey 350g",.22),("Rolled Oats 1kg","Bananas 1kg",.28),
("Chicken Breast 1kg","Basmati Rice 1kg",.38),("Chicken Thighs 1kg","Jasmine Rice 1kg",.32),("Chicken Breast 1kg","Garlic Bulbs 3pk",.32),("Whole Chicken 1.5kg","Potatoes 2kg",.30),("Chicken Thighs 1kg","Onions Yellow 1kg",.30),
("Black Tea Bags 100pk","Tea Biscuits 300g",.50),("Chai Tea 50pk","Digestive Biscuits 400g",.38),("Green Tea Bags 25pk","Honey 350g",.22),("Black Tea Bags 100pk","Whole Milk 1L",.35),
("White Bread Loaf","Salted Butter 250g",.42),("White Bread Loaf","Strawberry Jam 340g",.26),("White Bread Loaf","Cheddar Cheese 200g",.28),("Whole Wheat Bread Loaf","Sliced Ham 200g",.32),("Sliced Ham 200g","Cheddar Cheese 200g",.38),("Sliced Swiss Cheese 150g","Sliced Ham 200g",.33),
("Burger Buns 6pk","Beef Burger Patties 4pk",.55),("Burger Buns 6pk","Tomato Ketchup 500g",.40),("Hot Dog Rolls 6pk","Frankfurters 8pk",.58),("Hot Dog Rolls 6pk","Yellow Mustard 250g",.35),
("Potato Chips Salted","Cola 2L",.38),("Tortilla Chips 300g","Salsa Dip 300g",.52),("Tortilla Chips 300g","Guacamole 150g",.30),("Microwave Popcorn 3pk","Cola 2L",.28),
("Taco Shells 12pk","Ground Beef 500g",.50),("Taco Shells 12pk","Taco Seasoning",.62),("Flour Tortillas 8pk","Cheddar Cheese 200g",.30),
("Salmon Fillet 300g","Lemons 500g",.38),("Cod Fillet 300g","Lemons 500g",.30),("Raw Shrimp 400g","Garlic Bulbs 3pk",.36),
("Baby Diapers Size 3","Baby Wipes 3pk",.62),("Baby Diapers Size 4","Baby Wipes 3pk",.62),("Infant Formula 800g","Baby Diapers Size 3",.30),("Baby Cereal 250g","Fruit Puree Pouches 4pk",.40),
("Dry Dog Food 5kg","Dog Biscuits 500g",.32),("Wet Cat Food 12pk","Cat Litter 10L",.32),("Dry Cat Food 2kg","Cat Treats 60g",.26),
("Toothpaste 100ml","Toothbrush 2pk",.40),("Toothpaste 100ml","Mouthwash 500ml",.25),("Shampoo 400ml","Conditioner 400ml",.58),("Razor Blades 4pk","Shaving Foam 200ml",.52),
("Laundry Detergent 2L","Fabric Softener 1.5L",.46),("Dishwashing Liquid 750ml","Dish Sponges 6pk",.30),("Toilet Paper 12pk","Paper Towels 4pk",.40),
("Pancake Mix 500g","Maple Syrup 250ml",.58),("Eggs Large 12pk","Salted Butter 250g",.22),("Eggs Large 12pk","White Bread Loaf",.28),("Streaky Bacon 250g","Eggs Large 12pk",.52),
("Lager Beer 6pk","Potato Chips Salted",.32),("Red Wine 750ml","Parmesan Cheese 100g",.22),("Vanilla Ice Cream 1L","Ice Cream Cones 6pk",.22),("Chocolate Ice Cream 1L","Chocolate Syrup 350g",.18),
("Peanut Butter 340g","White Bread Loaf",.42),("Strawberry Jam 340g","Salted Butter 250g",.22),("Greek Yogurt 500g","Strawberries 400g",.22),
("Apples Red 1kg","Bananas 1kg",.32),("Bananas 1kg","Apples Red 1kg",.30),("Oranges 1kg","Apples Red 1kg",.26),("Oranges 1kg","Bananas 1kg",.22),
("Tomatoes 1kg","Onions Yellow 1kg",.36),("Tomatoes 1kg","Garlic Bulbs 3pk",.32),("Cucumbers 3pk","Tomatoes 1kg",.36),("Lettuce Iceberg","Tomatoes 1kg",.30),
("Basmati Rice 1kg","Garam Masala 100g",.16),("Basmati Rice 1kg","Turmeric Powder 100g",.16),("Red Lentils 1kg","Basmati Rice 1kg",.30),("Paneer 200g","Tomatoes 1kg",.30),("Ghee 500ml","Basmati Rice 1kg",.22),
("Rice Noodles 400g","Soy Sauce 250ml",.36),("Instant Noodles 5pk","Soy Sauce 250ml",.15),("Rice Noodles 400g","Sweet Chili Sauce 300ml",.22),
("Baking Powder 100g","All Purpose Flour 1kg",.55),("All Purpose Flour 1kg","White Sugar 1kg",.36),("Cocoa Powder 250g","Brown Sugar 500g",.22),("Vanilla Extract 50ml","Baking Powder 100g",.32),
("Birthday Candles 24pk","Cake Mix Vanilla",.40),("Paper Plates 50pk","Disposable Cups 50pk",.62),("Balloons 20pk","Birthday Candles 24pk",.30),
("Hummus 200g","Pita Bread 6pk",.52),("Tzatziki 200g","Pita Bread 6pk",.32),("Frozen Pepperoni Pizza","Cola 2L",.22),("Canned Tuna 3pk","Mayonnaise 400g",.32),
("Whole Milk 1L","Corn Flakes 500g",.16),("Whole Milk 1L","White Bread Loaf",.22),("Cheddar Cheese 200g","Cream Crackers 300g",.16),
]
CATW = {"Fresh Fruits":4,"Citrus Fruits":1.6,"Fresh Vegetables":3,"Herbs & Aromatics":3,"Milk":3.5,"Cheese":3.5,"Bread":3.5,"Eggs":2.5,"Chicken":3,"Rice":2.6,"Pasta":2.6,
        "Pasta Sauce":1.4,"Coffee":2.8,"Deli Ham & Cold Cuts":2.2,"Fish":2,"Sugar & Sweeteners":1.8,"Tea":1.8,"Butter & Margarine":1.8,"Soft Drinks":1.8,"Chips & Crisps":1.5,
        "Breakfast Cereal":1.6,"Cookies & Biscuits":1.6,"Root Vegetables":1.8,"Yogurt":1.6,"Cooking Oil":1.4,"Leafy Vegetables":1.5,"Toilet Paper":0}
BOOST = {"Bananas 1kg":2.0,"Cheddar Cheese 200g":1.8,"White Bread Loaf":1.8,"Sliced Ham 200g":1.6,"Apples Red 1kg":1.6,"Oranges 1kg":1.5,"Chicken Breast 1kg":1.5,"Basmati Rice 1kg":1.5,
         "Garlic Bulbs 3pk":1.8,"Spaghetti 500g":1.6,"Tomato Pasta Sauce 400g":1.0,"Ground Coffee 250g":1.4,"Salmon Fillet 300g":1.2,"White Sugar 1kg":1.5,"Whole Milk 1L":1.2}

rows = []
for line in CATALOG.strip().splitlines():
    cat, items = line.split("|")
    for it in items.split(";"):
        n, p = it.rsplit(":", 1); rows.append((cat, n.strip(), float(p)))
prod = pd.DataFrame(rows, columns=["Category", "Product", "Price"])
prod["ProductID"] = ["P%04d" % (i + 1) for i in range(len(prod))]
cats = list(dict.fromkeys(prod.Category)); name2i = {n: i for i, n in enumerate(prod.Product)}
print("categories:", len(cats), "products:", len(prod))
for a, b, p in AFF: assert a in name2i and b in name2i, (a, b)

pw = np.array([CATW.get(c, 1.0) * BOOST.get(n, 1.0) * rng.lognormal(0, .45) for c, n in zip(prod.Category, prod.Product)])
pcat = prod.Category.values
def mult(mask_cats, f, w): return np.where(np.isin(pcat, mask_cats), w * f, w)
BABY = ["Baby Food","Diapers & Wipes","Baby Formula","Baby Bath & Care"]; DOG = ["Dog Food"]; CAT = ["Cat Food","Pet Litter & Care"]
ALC = ["Beer","Wine","Spirits & Mixers"]; MEAT = ["Chicken","Beef","Pork","Lamb & Mutton","Deli Ham & Cold Cuts","Sausages & Bacon","Fish","Shrimp & Prawns","Shellfish","Frozen Meat & Nuggets","Frozen Seafood","Smoked & Cured"]
P = np.zeros((N_CUST, len(prod)))
for c in range(N_CUST):
    w = pw * rng.lognormal(0, .35, len(prod))
    fav = rng.choice(len(cats), 7, replace=False); w = w * np.where(np.isin(pcat, [cats[i] for i in fav]), 2.5, 1)
    w = mult(BABY, 8 if rng.random() < .08 else .03, w); w = mult(DOG, 8 if rng.random() < .13 else .03, w); w = mult(CAT, 8 if rng.random() < .10 else .03, w)
    w = mult(ALC, 3 if rng.random() < .22 else .03, w)
    if rng.random() < .10: w = mult(MEAT, .05, w)
    P[c] = w / w.sum()

days = pd.date_range("2023-01-01", "2024-12-31")
season = {1:.95,2:.9,3:.98,4:1.0,5:1.03,6:1.0,7:.97,8:.98,9:1.0,10:1.05,11:1.1,12:1.25}
dow = {0:.85,1:.8,2:.85,3:.95,4:1.1,5:1.4,6:1.25}
dw = np.array([season[d.month] * dow[d.dayofweek] * (1.08 if d.year == 2024 else 1.0) * (1 + .08 * np.sin(i / 11)) for i, d in enumerate(days)]); dw /= dw.sum()
tx_days = days[rng.choice(len(days), N_TX, p=dw)]
cw = rng.lognormal(0, .7, N_CUST); cw /= cw.sum()
tx_cust = np.concatenate([np.arange(N_CUST), rng.choice(N_CUST, N_TX - N_CUST, p=cw)]); rng.shuffle(tx_cust)
order = np.argsort(tx_days, kind="stable"); tx_days, tx_cust = tx_days[order], tx_cust[order]
aff = {}
for a, b, p in AFF: aff.setdefault(name2i[a], []).append((name2i[b], p))
QV = [1, 2, 3, 4]; QP = [.74, .19, .05, .02]
out = []
for t in range(N_TX):
    c = tx_cust[t]; n = int(np.clip(1 + rng.poisson(2.3), 1, 11))
    basket = set(rng.choice(len(prod), n, replace=False, p=P[c]).tolist())
    for a in list(basket):
        for b, p in aff.get(a, []):
            if rng.random() < p: basket.add(b)
    d = tx_days[t]; infl = 1.035 if d.year == 2024 else 1.0
    for i in sorted(basket)[:14]:
        q = int(rng.choice(QV, p=QP)); price = prod.Price[i] * infl * (0.9 if rng.random() < .1 else 1)
        price = round(price, 2)
        out.append(("T%05d" % (t + 1), "C%04d" % (c + 1), d.strftime("%Y-%m-%d"), prod.ProductID[i], prod.Product[i], prod.Category[i], q, price, round(q * price, 2)))
df = pd.DataFrame(out, columns=["TransactionID","CustomerID","Date","ProductID","Product","Category","Quantity","UnitPrice","TotalAmount"])
df.to_csv(OUT, index=False); print(df.shape, "->", OUT)
