from models.EGDiT import DM


def dm_raw(**kwargs):
    layer = 8
    return DM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196, **kwargs)

def dm_raw_nocross(**kwargs):
    layer = 8
    return DM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196, cross = False, **kwargs)

def dm_raw_crossonly(**kwargs):
    layer = 8
    return DM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196, end = False, **kwargs)

def dm_raw_crossmix(**kwargs):
    layer = 8
    return DM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196, end = False,con = True, **kwargs)


DM_models = {'IsoDiff': dm_raw, "DM-nocross" :dm_raw_nocross , "DM-crossonly" : dm_raw_crossonly, "DM-crossmix":dm_raw_crossmix
}

