# Feature Catalog

All features are computed from the **training** split only (no leakage) and persisted under `data/features/` as parquet tables. The same definitions are reused at prediction time via `load_feature_store()`.

| Entity | Feature | Type | Description |
|--------|---------|------|-------------|
| user | `user_avg_rating` | float | Mean rating the user has given across all training interactions. |
| user | `user_activity_count` | int | Number of interactions the user has in the training data. |
| user | `user_distinct_items` | int | Number of distinct items the user interacted with. |
| user | `user_rating_std` | float | Standard deviation of the user's ratings (rating consistency). |
| user | `user_days_since_last` | int | Days between the user's last interaction and the reference date. |
| user | `user_preferred_category` | str | Item category the user interacts with most often. |
| item | `item_popularity` | int | Number of interactions the item received (interaction count). |
| item | `item_distinct_users` | int | Number of distinct users who interacted with the item. |
| item | `item_avg_rating` | float | Mean rating the item received. |
| item | `item_rating_std` | float | Standard deviation of the item's ratings. |
| item | `item_release_year` | int | Release year of the item (from the item catalog). |
| item | `item_category` | str | Category of the item (from the item catalog). |
| interaction | `rating` | float | Explicit rating given in the interaction (target for RMSE/MAE). |
| interaction | `event_type` | str | Type of interaction: rating / view / click / purchase. |
| interaction | `interaction_month` | int | Calendar month of the interaction (1-12), a seasonality signal. |
| interaction | `interaction_dayofweek` | int | Day of week of the interaction (0=Mon..6=Sun). |
