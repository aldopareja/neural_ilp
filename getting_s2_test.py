with open('data/countries_s1') as s1:
	s1_facts = set(s1.read().splitlines())
with open('data/countries_s2') as s2:
	s2_facts = set(s2.read().splitlines())
with open('data/countries_s3') as s3:
	s3_facts = set(s3.read().splitlines())
with open('data/s1_test') as ts1:
	ts1_facts = set(ts1.read().splitlines())

#build test set for s2
ts2_facts = s1_facts.difference(s2_facts)
ts2_facts = ts1_facts.union(ts2_facts)
with open('data/s2_test','w') as ts2:
	ts2.writelines(['%s\n' % t for t in ts2_facts])

#build test set for s3
ts3_facts = s1_facts.difference(s3_facts)
ts3_facts = ts1_facts.union(ts3_facts)
with open('data/s3_test','w') as ts3:
	ts3.writelines(['%s\n' % t for t in ts3_facts])
