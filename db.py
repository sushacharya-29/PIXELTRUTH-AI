from sklearn.utils import resample
from supabase import create_client

url = "https://xnhwpspmipcnqcdfgyyp.supabase.co"
key = "sb_publishable_KSxoWK6KQgxfcofrKX-Sxg_TofsxGKt"

supabase = create_client(url, key)

# INSERT TEST
supabase.table("scans").insert({
    "result": "GAN"
}).execute()

print("Inserted!")

res= supabase.table("scans").select("*").execute()
print(res)