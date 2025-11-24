from django.db import migrations


def ensure_alias_table(apps, schema_editor):
    """
    Guard-rail migration: create SubjectTarget/SubjectAlias tables if they were
    missed in earlier deploys. Safe to run on databases where they already exist.
    """
    connection = schema_editor.connection
    existing = set(connection.introspection.table_names())

    SubjectTarget = apps.get_model("matcher", "SubjectTarget")
    SubjectAlias = apps.get_model("matcher", "SubjectAlias")

    created = []

    if SubjectTarget._meta.db_table not in existing:
        schema_editor.create_model(SubjectTarget)
        existing.add(SubjectTarget._meta.db_table)
        created.append(SubjectTarget._meta.db_table)

    if SubjectAlias._meta.db_table not in existing:
        schema_editor.create_model(SubjectAlias)
        created.append(SubjectAlias._meta.db_table)

    if created:
        # Emit to console so deploy logs show it happened; avoids a logger dependency here.
        print(f"[migration] created matcher tables: {', '.join(created)}")


class Migration(migrations.Migration):

    dependencies = [
        ("matcher", "0002_subjecttarget_coef"),
    ]

    operations = [
        migrations.RunPython(ensure_alias_table, migrations.RunPython.noop),
    ]
