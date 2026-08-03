.PHONY: test verify clean

test:
	pytest tests -q

verify:
	regbench verify

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache
