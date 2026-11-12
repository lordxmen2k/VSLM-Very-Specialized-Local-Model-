# Thin forwarder to the repository-root Makefile.
#
# `make preflight` should work whether you are at the repo root or inside a
# subdirectory. Make does not search upward for a makefile, so this file
# forwards every target to the one that does the work.
#
#     cd scripts && make preflight    # now works
#     make preflight                  # still works

ROOT := $(abspath $(dir $(firstword $(MAKEFILE_LIST)))..)

# Only the targets with their own recipe are listed here. Declaring the
# forwarded ones .PHONY would give them an empty explicit rule, and an
# explicit rule wins over the catch-all pattern below - so `make preflight`
# would report "Nothing to be done" instead of forwarding.

help all:
	$(MAKE) -C $(ROOT) $@

%:
	$(MAKE) -C $(ROOT) $@
