from django.db.models import FileField
from django.forms import forms
from django.template.defaultfilters import filesizeformat
from django.utils.translation import ugettext_lazy as _


class MyFileField(FileField):
    """
    Same as FileField, but you can specify:
        * content_types - list containing allowed content_types. Example: ['application/pdf', 'image/jpeg']
        * max_upload_size - a number indicating the maximum file size allowed for upload.
            2.5MB - 2621440
            5MB - 5242880
            10MB - 10485760
            20MB - 20971520
            50MB - 5242880
            100MB - 104857600
            250MB - 214958080
            500MB - 429916160
    """

    def __init__(self, *args, **kwargs):
        self.content_types = kwargs.pop("content_types", [])
        self.max_upload_size = kwargs.pop("max_upload_size", 0)

        super(MyFileField, self).__init__(*args, **kwargs)

    def clean(self, *args, **kwargs):
        data = super(MyFileField, self).clean(*args, **kwargs)

        file = data.file
        try:
            content_type = file.content_type
            print(f"->{content_type} {file.size}, {self.max_upload_size}")
            # print("Content type:", content_type, file._size, self.max_upload_size,)
            if len(self.content_types) == 0 or content_type in self.content_types:
                if file.size > self.max_upload_size:
                    raise forms.ValidationError(
                        _("Please keep filesize under %s. Current filesize %s")
                        % (
                            filesizeformat(self.max_upload_size),
                            filesizeformat(file.size),
                        )
                    )
            else:
                print(self.content_types, "hello")
                raise forms.ValidationError(_("Filetype not supported."))
        except AttributeError as e:
            print(e)
            pass

        return data
