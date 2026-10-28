# Publish RetailFlow v0.1.0

## What belongs in Git
Copy pyproject.toml, readme.md and environment/ from this kit into your repository,
review the diff, and commit/push them. The only dependency-range change allows
sentence-transformers 6.0.1, which is in your working environment. No model reload
is needed merely to apply these text changes. You may also keep release notes.

## What belongs in GitHub Release assets
Keep retailflow-models-v0.1.0.zip and retailflow-models-v0.1.0.sha256 outside your
source checkout. Do not git add the bundle or individual trained models.
Your working model folders already contain the same trained files; leave them
as they are. Discard the older rc1 download once you have the final download.

## Create the release in GitHub
Open LeonovYurii/RetailFlow, then Releases and Draft a new release (or Create a
new release). Choose a new tag v0.1.0 targeting the commit containing the updated
README and pyproject.toml. Title: RetailFlow v0.1.0.
Paste RELEASE_NOTES_v0.1.0.md into the release description.
Attach the final model ZIP and matching .sha256 file. Saving a draft lets you
review all files before publishing. GitHub will add source archives itself.

The README's direct download links activate only after that release is published.
Do not attach retailflow_release_files.zip: it is a maintainer merge kit.

## Distribution terms
The bundle includes upstream Llama/embedding notices. No license has been chosen
on your behalf for your original source or classifier. Specify their distribution
terms and confirm relevant training-data rights before public distribution.
Keep the included model notices and the Built with Llama attribution.

## After publication
Check that both release-asset links work and that the source tag contains the
dependency correction. A third party downloads source + model bundle, supplies
access to the upstream base model, installs the environment, and launches chat.
Nothing in this preparation process pushes, tags, or publishes to GitHub.
