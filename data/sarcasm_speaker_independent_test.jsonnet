std.prune(std.mapWithKey(function(id, x) if x.show == 'FRIENDS' then x else null, import 'sarcasm_data.json'))
