std.prune(std.mapWithKey(function(id, x) if x.show == 'FRIENDS' then null else x, import 'sarcasm_data.json'))
