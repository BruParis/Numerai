# need to load features order
FEATURES_ORDER = ['feature_intelligence', 'feature_charisma', 'feature_strength',
                  'feature_dexterity', 'feature_constitution', 'feature_wisdom']

# need to make sure cat_split_idx > num features
CATEGORY_SPLIT_IDX = 10000


def ft_category_idx(ft):
    for i in range(len(FEATURES_ORDER)):
        if FEATURES_ORDER[i] in ft:
            return i


def ft_cat_order_idx(ft):
    cat = ft_category_idx(ft)
    idx = cat * CATEGORY_SPLIT_IDX + int(ft.replace(FEATURES_ORDER[cat], ''))
    return idx
