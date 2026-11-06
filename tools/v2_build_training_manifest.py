from pathlib import PurePosixPath


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe dataset-relative path')
    return str(path)


def assemble(shards, layout, expected_hosts):
    if set(shards) != set(expected_hosts) or layout['scope'] != 'full_verified_dataset':
        raise ValueError('Training assembly requires all verified hosts and complete layout')
    expected_layout = {source['manifest_sha256'] for source in layout['source_shards']}
    actual = {shard['manifest_sha256'] for shard in shards.values()}
    if expected_layout != actual or len(actual) != len(shards):
        raise ValueError('Layout does not cover exactly the verified training shards')
    vocabulary = None
    entries, files, lineage, seen = [], {}, [], set()
    for host in sorted(shards):
        shard = shards[host]
        manifest, base = shard['manifest'], shard['base_manifest']
        vocab = base['schema']['vocab']
        if vocabulary is None:
            vocabulary = vocab
        elif vocab != vocabulary:
            raise ValueError('Source shard vocabularies differ; explicit aligned rematerialization required')
        views = {}
        for view in base['entries']:
            key = (view['tag'], view['owner'])
            if key in views:
                raise ValueError('Duplicate base player view')
            views[key] = view
        for episode in manifest['entries']:
            tag = episode['tag']
            if tag in seen:
                raise ValueError('Duplicate episode across hosts')
            seen.add(tag)
            episode_name = safe_relative('episodes/'+host+'/'+safe_relative(episode['file']))
            identity = dict(sha256=episode['sha256'], bytes=episode['bytes'], source_host=host,
                            source_path=safe_relative(episode['file']))
            files[episode_name] = identity
            base_file = None
            for owner in (0, 1):
                if (tag, owner) not in views:
                    raise ValueError('Accepted episode lacks both frozen player views')
                view = views[(tag, owner)]
                if view['validation'] != episode['validation'] or view['decisions'] != episode['frames']:
                    raise ValueError('Frozen split or decision count differs from episode')
                source_file = safe_relative(view['file'])
                if base_file is not None and base_file != source_file:
                    raise ValueError('Both owners must reference the same archive')
                base_file = source_file
                delivered_base = safe_relative('base/'+host+'/'+source_file)
                entries.append(dict(file=episode_name, sha256=episode['sha256'], base_archive=delivered_base,
                                    decisions=episode['frames'], tag=tag, owner=owner,
                                    validation=episode['validation'], source_host=host))
            base_identity = shard['base_files'].get(base_file)
            if not base_identity or not base_identity.get('sha256') or not base_identity.get('bytes'):
                raise ValueError('Frozen archive delivery lacks independently checked identity')
            files[delivered_base] = dict(base_identity, source_host=host, source_path=base_file)
        lineage.append(dict(host=host, manifest_sha256=shard['manifest_sha256'],
                            verified_episodes=len(manifest['entries']), excluded=len(manifest.get('failed', []))))
    if len(seen) != layout['episodes']:
        raise ValueError('Merged layout episode count differs from training scope')
    return dict(input_profile='inputs-v2-history31-own2',
                schema=dict(vocab=vocabulary, inputs=2, label_policy='frozen source arrays; no relabeling'),
                entries=entries, lineage=lineage), dict(files=files)
