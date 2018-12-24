params={
'symbols_in_batch':1000,
'num_hidden':600,
'attnsize':600,
'epochs':100,
'max_seqlen':1000,
'embed_size':200,
'symbols':8000,
'filter_width':3,
'net_verbose':False,
'batchnorm':False, #True causes training failure, gradient clipping might fix this
'upsample':False,
'weight_decay':3e-4,
'heads':8,
}
